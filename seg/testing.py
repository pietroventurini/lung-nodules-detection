import argparse
import datetime
import os
import glob

from torch.utils.data import DataLoader
from seg.dsets import Luna2dSegmentationDataset
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim



from util.util import enumerateWithEstimate
from util.logconf import logging
from .model import UNetWrapper

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


class TestingSegmentation():
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

        parser.add_argument('--tb-prefix',
            default='seg',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='unibs',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.seg_model = None


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

        prediction_g = self.seg_model(input_g) # prediction is (batch_size, 1, 512, 512)

        # normal dice loss for the samples in our dataset
        diceLoss_g = self.diceLoss(prediction_g, label_g) 
        # dice loss for only the pixels included in label_g (only false negatives will generate loss)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g) 

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            # convert to float for later multiplication
            # 0:1 accounts for the unique element along the index dimension (mask is 2D)
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32) 

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
        
        params
        -------
            prediction_g : tensor
            A tensor of shape (batch_size, 1, 512, 512) containing
            the predicted segmentation mask for the corresponding CT slice
            label_g : tensor
            A tensor of shape (batch_size, 1, 512, 512) containing
            the target segmentation mask for the corresponding CT slice
        """
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g
    
    def main(self):

        log.info("Evaluate performances on test set")

        test_dl = self.initTestDl()

        # load best model according to performances on validation set
        segmentation_path = self.getBestModelPath('seg')
        log.debug(segmentation_path)
        seg_dict = torch.load(segmentation_path, map_location=self.device)

        self.seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        self.seg_model.load_state_dict(seg_dict['model_state'])
        self.seg_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                self.seg_model = nn.DataParallel(self.seg_model)
            self.seg_model.to(self.device) # move model to GPU

        # tensor containing per-sample metrics
        testMetrics_g = torch.zeros(METRICS_SIZE, len(test_dl.dataset), device=self.device)

        batch_iter = enumerateWithEstimate(
            test_dl,
            "Testing",
            start_ndx=test_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.computeBatchLoss(batch_ndx, batch_tup, test_dl.batch_size, testMetrics_g)
            
        metrics_a = testMetrics_g.to('cpu').detach().numpy()
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

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

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


if __name__ == '__main__':
    TestingSegmentation().main()