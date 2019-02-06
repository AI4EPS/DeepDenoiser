### Related paper:
Zhu, W., Mousavi, S. M., & Beroza, G. C. (2018). Seismic Signal Denoising and Decomposition Using Deep Neural Networks. arXiv preprint [arXiv:1811.02695](https://arxiv.org/abs/1811.02695).

## Install
The code is tested under python3.6
```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo
### 1.Data
Located in directory: **Dataset**

Data format:

1. A csv file of all the input file names

2. A directory of the actually numpy files. The .npz file has one field called "data" which is the waveform data following a format of "3000x1" with sampling rate of 100 Hz. 

### 2.Model
Located in directory: **log/1106095243**
### 3. Prediction
~~~python
python run_queue.py --mode=pred --ckdir=log/1106095243 --batch_size=10
~~~

### 4. Output
Located in directory: **log/pred/**

## Hyper paramters
~~~
python run_queue.py --help
~~~
```bash
usage: run_queue.py [-h] [--mode MODE] [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                    [--decay_step DECAY_STEP] [--decay_rate DECAY_RATE]
                    [--momentum MOMENTUM] [--filters_root FILTERS_ROOT]
                    [--depth DEPTH]
                    [--kernel_size KERNEL_SIZE [KERNEL_SIZE ...]]
                    [--pool_size POOL_SIZE [POOL_SIZE ...]]
                    [--drop_rate DROP_RATE]
                    [--dilation_rate DILATION_RATE [DILATION_RATE ...]]
                    [--loss_type LOSS_TYPE] [--weight_decay WEIGHT_DECAY]
                    [--optimizer OPTIMIZER] [--summary SUMMARY]
                    [--class_weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]]
                    [--logdir LOGDIR] [--ckdir CKDIR]
                    [--plot_number PLOT_NUMBER] [--fpred FPRED]
                    [--plot_pred PLOT_PRED] [--save_pred SAVE_PRED]
```