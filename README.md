### Related paper:
Zhu, W., Mousavi, S. M., & Beroza, G. C. (2018). Seismic Signal Denoising and Decomposition Using Deep Neural Networks. arXiv preprint [arXiv:1811.02695](https://arxiv.org/abs/1811.02695).

## 1. Install
The code is tested under python3.6

### Using virtualenv
```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Using ananconda
```bash
conda create --name venv python=3.6
conda activate venv
conda install tensorflow=1.10 matplotlib scipy pandas tqdm
```

## 2. Demo data

Located in directory: **Dataset/**

## 3. Model

Located in directory: **Model/190624-234436**

## 4. Prediction
### Data format:

Required a csv file and a directory of npz files.

The csv file contains one column: "fname"

The npz file contains one variable: "data"

The shape of "data" variable has a shape of 3000 x 3 with sampling rate of 100 Hz. 

~~~bash
python run.py --mode=pred --model_dir=Model/190624-234436 --data_dir=./Dataset/pred --data_list=./Dataset/pred.csv --output_dir=./output --plot_figure --save_result --batch_size=20
~~~

Notes:

1. For large dataset and GPUs, larger batch size can accelerate the prediction.
2. Plotting figures is slow. Removing the argument of --plot_figure can speed the prediction

## 5. Train
### Data format

Required a csv file and a directory of npz files.

The csv file contains four columns: "fname", "itp", "channels"

The npz file contains four variable: "data", "itp",  "channels"

The shape of "data" variables has a shape of 9001 x 3

The variables "itp" is the data points of first P arrival times. 

~~~bash
python run.py --mode=train --data_dir=./Dataset/train --data_list=./Dataset/train.csv --batch_size=20
~~~

Please let us know of any bugs found in the code. Suggestions and collaborations are welcomed!