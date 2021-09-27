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

raw_cache = getCache('cls_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz',
)
MaskTuple = namedtuple(
    'MaskTuple',
    'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask',
)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    """ Construct a set with all series_uids that are present on disk.
    This will let us use the data, even if we haven't downloaded all of
    the subsets yet.
    """
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    # loop over nodules
    with open('data/part2/luna/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]: # for each line that represents a nodule
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
                    annotationCenter_xyz
                )
            )

   # loop over candidates (we are keeping only non-nodules) 
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool: # only non-nodules
                candidateInfo_list.append(CandidateInfoTuple(
                    False, # isNodule_bool
                    False, # hasAnnotation_bool
                    False, # isMal_bool
                    0.0,
                    series_uid,
                    candidateCenter_xyz,
                ))

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
    """ Represents a single CT scan together with its series_uid and metadata
    """
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

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

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    """ Caches the ct_chunk and the mask (pos_chunk) 
    """
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    """ Caches the size of each ct scan. It helps to quickly construct
    the full validation set without having to load each Ct at Dataset
    initialization.
    """
    ct = Ct(series_uid, buildMasks_bool=False)
    return len(ct.negative_indexes)

def getCtAugmentedCandidate(augmentation_dict, series_uid, center_xyz, width_irc, use_cache=True):
    """ Take a chunk-of-CT-with-candidate-inside and modify it by
    applying transformations defined in the `augmentation_dict`, such as
    mirroring, shifting, scaling, rotating and noise addition. 
    """
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1) # in [-1,1]
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1) # in [-1,1]
            transform_t[i,i] *= 1.0 + scale_float * random_float


    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        # rotation on the X-Y subspace (ignore Z because scaling along the 
        # index-axis is different than scaling along rows or cols axis)
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    # compute the flow-field grid of size (N, H_out, W_out, 2) such that
    # for grid[n, h, w] specifies the input pixel location x and y that will
    # be used to interpolate the output value output[n, :, h, w]
    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            ct_t.size(),
            align_corners=False,
        )

    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border', # use border values for out-of-bound grid locations
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    """ Represents the dataset for the nodule-nonnodule classifier"""
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 isTestSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 candidateInfo_list=None,
            ):
        """ Initialize a dataset of candidate nodules.
        Parameters
        -------
        val_stride : int
            The ratio of training data to validation/testing data.
            E.g. if val_stride = 10, then pick a validation (or testing) sample 
            every 10 samples in the dataset. For the validation set we keep the
            1st, 11th, 21st... examples, while for the test set we keep the 2nd,
            12th, 22nd... examples.

        isValSet_bool : bool
            Indicates whether we are instantiating a dataset for validation or not.

        isTestSet_bool : bool
            Indicates whether we are instantiating a dataset for testing or not.

        ratio_int : int
            Tatio of negative to positive samples.
            E.g., `ratio_int=1` means 1:1 ratio of positive-to-negative examples.

        sortby_str : String
            Sorting criterion (`random`, `series_uid`, `label_and_size`).
            
        augmentation_dict : Dictionary
            The augmentation techniques to apply to each sample: elements can be
            {'flip':bool, 'offset':float, 'scale':float, 'rotate':bool, 'noise':float}.
        """
        if (isValSet_bool and isTestSet_bool):
            raise Exception("At most one between isValSet_bool and isTestSet_bool must be provided")
        
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(candidateInfo_tup.series_uid for candidateInfo_tup in self.candidateInfo_list))

        if isValSet_bool:
            # Starting with a series list containing all our series, 
            # we keep only every val_stride-th element, starting with 0.
            assert val_stride > 0, val_stride
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

        series_set = set(self.series_list) # set for faster lookup
        # filter out candidates from series not in our set
        self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid in series_set]

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.neg_list = [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]
        self.ben_list = [nt for nt in self.pos_list if not nt.isMal_bool]
        self.mal_list = [nt for nt in self.pos_list if nt.isMal_bool]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.candidateInfo_list)
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    def __len__(self):
        if self.ratio_int:
            return 50000 # we are no tied to a specific number of samples if we present positive samples repeatedly
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        """ Returns the ndx nodule in the dataset performing
        balancing of pos/neg samples (only for the training set)
        """
        if self.ratio_int: # only for training set
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1): # non-zero remainder => negative sample
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidateInfo_tup = self.neg_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list) # avoid exhausting the dataset when using a low ratio_int
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:  # in case we are not balancing classes (validation, test)
            candidateInfo_tup = self.candidateInfo_list[ndx]

        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isNodule_bool
        )

    def sampleFromCandidateInfo_tup(self, candidateInfo_tup, label_bool):
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        label_t = torch.tensor([False, False], dtype=torch.long)

        if not label_bool:
            label_t[0] = True
            index_t = 0
        else:
            label_t[1] = True
            index_t = 1

        return candidate_t, label_t, index_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)


class MalignantLunaDataset(LunaDataset):
    """ Represents the Dataset for the malignancy classifier.
    """
    def __len__(self):
        if self.ratio_int:
            return 100000
        else:
            return len(self.ben_list + self.mal_list)

    def __getitem__(self, ndx):
        """ Returns the ndx nodule in the dataset performing
        balancing of ben/mal samples (only for the training set)
        """
    
        """ # try with this solution:
        if self.ratio_int:
            if ndx % (self.ratio_int + 1):
                candidateInfo_tup = self.mal_list[(ndx // (self.ratio_int + 1)) % len(self.mal_list)]
            else:
                candidateInfo_tup = self.ben_list[(ndx // (self.ratio_int + 1)) % len(self.ben_list)]
        """
        if self.ratio_int: # only for training set
            if ndx % 2 != 0: # sample from mal_list (malignant nodules with frequency = 1/2)
                candidateInfo_tup = self.mal_list[(ndx // 2) % len(self.mal_list)]
            elif ndx % 4 == 0: # sample from ben_list (benign nodules with frequency 1/4)
                candidateInfo_tup = self.ben_list[(ndx // 4) % len(self.ben_list)]
            else: # sample_from neg_list (non-nodules with frequency 1/4)
                candidateInfo_tup = self.neg_list[(ndx // 4) % len(self.neg_list)]
        else: # in case we are not balancing mal/ben samples
            if ndx >= len(self.ben_list):
                candidateInfo_tup = self.mal_list[ndx - len(self.ben_list)]
            else:
                candidateInfo_tup = self.ben_list[ndx]

        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isMal_bool
        )
