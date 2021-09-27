# Deep learning project
## Detecting and classifying malignant lung nodules from CT scans using PyTorch

Refer to the [project report](Project&#32;Report.pdf) for the full description of the project.

This project is based on the book [Deep Learning with PyTorch by Eli Stevens, Luca Antiga, and Thomas Viehmann](https://www.manning.com/books/deep-learning-with-pytorch).

In the project directory you can find:

- `Training_models.ipynb`: notebook containing scripts to generate some of the figures in the report
and to train/test the models, as well as to plot metrics with TensorBoard.
- `requirements.txt`: requirements to create a VirtualEnv environment
- `download_dataset.sh`: script to download the datasets (120 GB) from the Internet
- `notes.md`: contains some logging messages from the training/testing of the models

The code can be found under the following directories:
- `seg`: code to train the segmentation model
- `cls`: code to train the nodule-nonnodule classifier and the malignancy classifier
- `util`: utility functions (e.g. logging, transformations, augmentation, cache...)

Data can be found/downloaded in the directories:
- `data/part2/luna/`: csv files containing annotated data about nodules and CT scans
- `data/part2/models/`: Pre-trained models (best models obtained during the trials)
- `data-unversioned/part2/luna/`: LUNA dataset (120 GB) which have to be downloaded from Internet
- `data-unversioned/part2/models/`: Trained models which have been saved (both best models and checkpoints)

Other directories:
- `runs/`: Tensorboard data
