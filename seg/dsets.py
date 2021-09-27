import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('seg_raw')

MaskTuple = namedtuple(
    'MaskTuple', 
    'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask'
)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', 
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []

    # loop over nodules from new annotation files 
    with open('data/part2/luna/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]: # only nodules
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool: # skip if not present on disk (pietro)
                continue

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True, # isNodule_bool
                    True, # hasAnnotation_bool
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool: # only non-nodules
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False, # isNodule_bool
                        False, # hasAnnotation_bool
                        False, # isMal_bool
                        0.0,
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    """Returns a dict {series_uid -> list_of_candidateInfo_tup}
    for each series_uid. Defaults to an empty list for those series_uids
    that do not have any candidateInfo_tup associated.
    """
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(candidateInfo_tup)

    return candidateInfo_dict

class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool
        ]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2)) 
                                 .nonzero()[0].tolist()) # Takes indices of the mask slices that have a nonzero count, which we make into a list

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)
        # loop over nodules
        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError: # in case we fall of the boundary of the tensor
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True
        # restrict mask to voxels above density threshold
        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        """ Crop a chunk of the Ct centered around `center_xyz` with 
        dimensions equal to `width_irc`.
        """
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)] # chunk of the ct scan
        pos_chunk = self.positive_mask[tuple(slice_list)] # chunk of the segmentation mask

        return ct_chunk, pos_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    """ Caches the ct_chunk and the mask (pos_chunk) """
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    """ Caches the size of each ct scan (since they can differ in size). 
    It helps to quickly construct the full validation set without having 
    to load each Ct at Dataset initialization.
    """
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 isTestSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False
                ):
        """
        Initialize a Luna2dSegmentationDataset: a base Dataset used
        for validation. The training set will extend this class. Consists of a 
        set of CT slices containing at least one positive example. By invoking
        __getitem___, it returns also contextSlices_count slices below and above
        that slice in the full CT scan.

        Parameters
        ----------
        val_stride : int
            The ratio of training data to validation/testing data.
            E.g. if val_stride = 10, then pick a validation (or testing) sample 
            every 10 samples in the dataset. For the validation set we keep the
            1st, 11th, 21st... examples, while for the test set we keep the 2nd,
            12th, 22nd... examples.

        isValSet_bool : bool
            Indicates whether we are instantiating a dataset for validation or not.
            If so, we will pick only every val_stride-th CTs

        isValSet_bool : bool
            Indicates whether we are instantiating a dataset for testing or not.
            If so, we will pick only every val_stride-th CTs

        contextSlices_count : int
            How many slices to consider in each direction along the index axis.
            E.g., if contextSlices_count = 3, we will consider 7 slices: the current one
            the 3 slices immediately before it and the 3 slices immediately after it.

        fullCT_bool : bool
            When fullCt_bool=True, we use every slice in the CT (useful for
            evaluating end-to-end performances, since we don't have any prior
            information). When fullCt_bool=False we consider only the slices that
            have a positive mask (useful during training)
        """
        if (isValSet_bool and isTestSet_bool):
            raise Exception("At most one between isValSet_bool and isTestSet_bool must be provided")
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            # Starting with a series list containing all our series, 
            # we keep only every val_stride-th element, starting with 0.
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif isTestSet_bool: # handle test set (same dimension of validation set)
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[1::val_stride]
        elif val_stride > 0:
            # if we are training, we delete every val_stride-th element.
            del self.series_list[::val_stride]
            # delete also samples in the test set (every val_stride-1 elements)
            del self.series_list[::val_stride-1]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indexes]

        self.candidateInfo_list = getCandidateInfoList() # cached

        series_set = set(self.series_list) # set for faster lookup
        # filter out candidates from series not in our set
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set] 

        # a list of actual nodules for the data balancing yet to come
        self.pos_list = [nt for nt in self.candidateInfo_list
                            if nt.isNodule_bool]

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        """ Returns the ndx-th element of the dataset, which is a slice of a CT scan,
        together with the contextSlices_count slices above and below it.
        """
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)] # wraps index around sample_list
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512)) # preallocates the output

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0) # if we reach beyond bounds, duplicate first slice
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1) # if we reach beyond bounds, duplicate last slice 
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    """ The training segmentation dataset inherits from Luna2dSegmentationDataset.
    Differently from the validation set, when training we want to crop slices around
    the nodules, in order to reduce the size of the input.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        """Returns a cropped region of the CT containing the nodule at the
        ndx-th position in pos_list.
        """
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)] # sample from pos_list instead of sample_list
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        """Returns a patch of the slice containing the candidate nodule
        of size 64x64 taken randomly from a 96x96 crop centered on the nodule.
        """
        ct_a, pos_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96),
        )
        pos_a = pos_a[3:4] # take one-element slice of the mask (keeps 3rd dimension, i.e., the single output channel)

        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)
        # crop a random 64x64 area from the 96x96 patch of the CT slice (which consists of 7 channels)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.float32)
        # crop 64x64 area of segmentation mask (which has been shrunk to 1 channel depth)            
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx

class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtRawCandidate(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96))

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1 #candidate_t, pos_t, series_uid, center_t