import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys
import glob

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from util.logconf import logging
from .model import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=16,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--tb-prefix',
            default='seg',
            help="Data prefix to use for Tensorboard run.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='unibs',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        # instantiate the segmentation model
        segmentation_model = UNetWrapper(
            in_channels=7, # 3+3 context slices + the current slice
            n_classes=1,
            depth=3, # 1 for each downsampling operation
            wf=4, 
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        # instantiate the augmentation model
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda: # move models to GPU
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def initOptimizer(self):
        return Adam(self.segmentation_model.parameters())
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)


    def initTrainDl(self):
        """ Initialize the training data loader. num_workers allows to
        loading data in parallel by using separate processes and shared memory. 
        Each worker produces a complete batch. When we iterate 
        (e.g. for batch_tup in self.train_dl) we won't have to wait for each Ct
        to be loaded, samples to be taken and so on, but we get the already loaded
        batch_tup immediately.
        """
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTestDl(self):
        test_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isTestSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return test_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '_val_seg_' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        self.validation_cadence = 5
        # loop over epochs
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted (first iteration and then every validation_cadence iterations)
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()
        self.evaluateOnTestSet()

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        """ Compute `diceLoss_g.mean() + fnLoss_g.mean() * 8`
        where diceLoss_g is the dice loss for the training samples
        and fnLoss is the dice loss only for the pixels included in label_g.
        The result is a weighted average in which we consider the entire population
        of our positive pixels right to be eight times more important than getting 
        the entire population of negative pixels right.
        """
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)

        # normal dice loss for the samples in our dataset
        diceLoss_g = self.diceLoss(prediction_g, label_g) 
        # dice loss for only the pixels included in label_g (only false negatives will generate loss)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g) 

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32) # convert to float for later multiplication

            tp = (     predictionBool_g * label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1,2,3])
            fp = (     predictionBool_g * (~label_g)).sum(dim=[1,2,3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        """ Compute dice loss (or Sørensen-Dice coefficient) as the
        per-pixel F1 score. It is defined as:
        2 * |X ∩ Y| / (|X|+|Y|)
        Since we want to solve a minimization problem, we return 1 - loss.
        """
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g

    def evaluateOnTestSet(self):

        log.info("Evaluate performances on test set")

        test_dl = self.initTestDl()

        # load best model according to performances on validation set
        segmentation_path = self.getBestModelPath('seg')
        log.debug(segmentation_path)
        seg_dict = torch.load(segmentation_path)

        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
            seg_model.to(self.device) # move model to GPU

        testMetrics_g = torch.zeros(METRICS_SIZE, len(test_dl.dataset), device=self.device)
        batch_iter = enumerateWithEstimate(
            test_dl,
            "Testing",
            start_ndx=test_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.computeBatchLoss(batch_ndx, batch_tup, test_dl.batch_size, testMetrics_g)
            
        testMetrics_t = testMetrics_g.to('cpu')
        metrics_a = testMetrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        # false positives can be > 100% because we're comparing the total number of pixels
        # labeled as candidate nodule, which is a tiny fraction of each image
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(("Test set performances "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            **metrics_dict,
        ))
        log.info(("Test set performances "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            **metrics_dict,
        ))

    
    def getBestModelPath(self, type_str):
        """ Retrieve the path to the most recently trained `type_str` model.
        """
        local_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            type_str + '_{}_{}.{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list: # use pre-trained model
            pretrained_path = os.path.join(
                'data',
                'part2',
                'models',
                type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise



    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval() # set model to eval

        images = sorted(dl.dataset.series_list)[:12] # take 12 CTs
        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid)

            for slice_ndx in range(6):
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5 # select 6 equidistant slices throughout the CT
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                ct_t, label_t, series_uid, ct_ndx = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1,:,:] /= 2000
                ct_t[:-1,:,:] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                image_a[:,:,0] += prediction_a & (1 - label_a) # false positives (red)
                image_a[:,:,0] += (1 - prediction_a) & label_a # false negatives (orange)
                image_a[:,:,1] += ((1 - prediction_a) & label_a) * 0.5 # false negatives (orange)

                image_a[:,:,1] += prediction_a & label_a # true positives (green)
                image_a *= 0.5 # image_a was in range [0,2], so we scale back to [0,1]
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC', # order of axes in our image
                )

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                    # image_a[:,:,0] += (1 - label_a) & lung_a # Red
                    image_a[:,:,1] += label_a  # Green
                    # image_a[:,:,2] += neg_a  # Blue

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC',
                    )
                # This flush prevents TB from getting confused about which
                # data item belongs where.
                writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        """ Log per-epoch metrics.
        params
        -----------
        epoch_ndx : int
            The index of the epoch to be logged
        mode_str : string
            A string like 'trn', 'val', 'test'
        metrics_t : array
            An array of shape (METRICS_SIZE, len(dataset)) containing metrics
            such as loss, tp, fn, fp accumulated during training/validation
        """
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        # false positives can be > 100% because we're comparing the total number of pixels
        # labeled as candidate nodule, which is a tiny fraction of each image
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score


    def saveModel(self, type_str, epoch_ndx, isBest=False):
        """ Save model parameters to disk. This is more flexible than
        saving the entire model, since we can reuse the parameters into any model
        that expects parameters of the same shape, even if the class doesn't match
        the model those parameters were saved under.
        """
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module # get rid of DataParallel wrapper (if it exists)

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(), # get model parameters
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(), # preserves momentum and so on (in order to resume training seamlessly)
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        # save a copy of the best model
        if isBest:
            best_path = os.path.join(
                'data-unversioned', 
                'part2', 
                'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    SegmentationTrainingApp().main()
