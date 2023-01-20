# DeepDenoiser: Seismic Signal Denoising and Decomposition Using Deep Neural Networks

[![](https://github.com/AI4EPS/DeepDenoiser/workflows/documentation/badge.svg)](https://ai4eps.github.io/DeepDenoiser)
## 1.  Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and requirements
- Download DeepDenoiser repository
```bash
git clone https://github.com/wayneweiqiang/DeeoDenoiser.git
cd DeepDenoiser
```
- Install to default environment
```bash
conda env update -f=env.yml -n base
```
- Install to "deepdenoiser" virtual envirionment
```bash
conda env create -f env.yml
conda activate deepdenoiser
```

## 2. Pre-trained model
Located in directory: **model/190614-104802**

## 3. Related papers
- Zhu, Weiqiang, S. Mostafa Mousavi, and Gregory C. Beroza. "Seismic Signal Denoising and Decomposition Using Deep Neural Networks." arXiv preprint arXiv:1811.02695 (2018).

## 4. Interactive example
See details in the [notebook](https://github.com/wayneweiqiang/DeepDenoiser/blob/master/docs/example_interactive.ipynb): [example_interactive.ipynb](example_interactive.ipynb)


## 5. Batch prediction
See details in the [notebook](https://github.com/wayneweiqiang/DeepDenoiser/blob/master/docs/example_batch_prediction.ipynb): [example_batch_prediction.ipynb](example_batch_prediction.ipynb)
## 6. Train
### Data format

Required: two csv files for signal and noise, corresponding directories of the npz files.

The csv file contains four columns: "fname", "itp", "channels"

The npz file contains four variable: "data", "itp",  "channels"

The shape of "data" variables has a shape of 9001 x 3

The variables "itp" is the data points of first P arrival times.

Note: In the demo data, for simplicity we use the waveform before itp as noise samples, so the train_noise_list is same as train_signal_list here.

~~~bash
python deepdenoiser/train.py --mode=train --train_signal_dir=./Dataset/train --train_signal_list=./Dataset/train.csv --train_noise_dir=./Dataset/train --train_noise_list=./Dataset/train.csv --batch_size=20
~~~

Please let us know of any bugs found in the code. Suggestions and collaborations are welcomed
