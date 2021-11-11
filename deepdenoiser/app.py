import os
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from json import dumps
from typing import Any, AnyStr, Dict, List, NamedTuple, Union

import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI
from kafka import KafkaProducer
from pydantic import BaseModel
import scipy
from scipy.interpolate import interp1d

from model import UNet

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

app = FastAPI()
X_SHAPE = [3000, 1, 3]
SAMPLING_RATE = 100

# load model
model = UNet(mode="pred")
sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=sess_config)
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
latest_check_point = tf.train.latest_checkpoint(f"{PROJECT_ROOT}/model/190614-104802")
print(f"restoring model {latest_check_point}")
saver.restore(sess, latest_check_point)

# Kafak producer
use_kafka = False
# BROKER_URL = 'localhost:9092'
# BROKER_URL = 'my-kafka-headless:9092'

try:
    print("Connecting to k8s kafka")
    BROKER_URL = "quakeflow-kafka-headless:9092"
    producer = KafkaProducer(
        bootstrap_servers=[BROKER_URL],
        key_serializer=lambda x: dumps(x).encode("utf-8"),
        value_serializer=lambda x: dumps(x).encode("utf-8"),
    )
    use_kafka = True
    print("k8s kafka connection success!")
except BaseException:
    print("k8s Kafka connection error")
    try:
        print("Connecting to local kafka")
        producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            key_serializer=lambda x: dumps(x).encode("utf-8"),
            value_serializer=lambda x: dumps(x).encode("utf-8"),
        )
        use_kafka = True
        print("local kafka connection success!")
    except BaseException:
        print("local Kafka connection error")


def normalize_batch(data, window=200):
    """
    data: nbn, nf, nt, 2
    """
    assert len(data.shape) == 4
    shift = window // 2
    nbt, nf, nt, nimg = data.shape

    ## std in slide windows
    data_pad = np.pad(data, ((0, 0), (0, 0), (window // 2, window // 2), (0, 0)), mode="reflect")
    t = np.arange(0, nt + shift - 1, shift, dtype="int")  # 201 => 0, 100, 200
    # print(f"nt = {nt}, nt+window//2 = {nt+window//2}")
    std = np.zeros([nbt, len(t)])
    mean = np.zeros([nbt, len(t)])
    for i in range(std.shape[1]):
        std[:, i] = np.std(data_pad[:, :, i * shift : i * shift + window, :], axis=(1, 2, 3))
        mean[:, i] = np.mean(data_pad[:, :, i * shift : i * shift + window, :], axis=(1, 2, 3))

    std[:, -1], mean[:, -1] = std[:, -2], mean[:, -2]
    std[:, 0], mean[:, 0] = std[:, 1], mean[:, 1]

    ## normalize data with interplated std
    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, kind="slinear")(t_interp)
    std_interp[std_interp == 0] = 1.0
    mean_interp = interp1d(t, mean, kind="slinear")(t_interp)

    data = (data - mean_interp[:, np.newaxis, :, np.newaxis]) / std_interp[:, np.newaxis, :, np.newaxis]

    if len(t) > 3:  ##need to address this normalization issue in training
        data /= 2.0

    return data


def get_prediction(meta):

    FS = 100
    NPERSEG = 30
    NFFT = 60

    vec = np.array(meta.vec)  # [batch, nt, chn]
    nbt, nt, nch = vec.shape
    vec = np.transpose(vec, [0, 2, 1])  # [batch, chn, nt]
    vec = np.reshape(vec, [nbt * nch, nt])  ## [batch * chn, nt]

    if np.mod(vec.shape[-1], 3000) == 1:  # 3001=>3000
        vec = vec[..., :-1]

    if meta.dt != 0.01:
        t = np.linspace(0, 1, len(vec))
        t_interp = np.linspace(0, 1, np.int(np.around(len(vec) * meta.dt * FS)))
        vec = interp1d(t, vec, kind="slinear")(t_interp)

    # sos = scipy.signal.butter(4, 0.1, 'high', fs=100, output='sos')  ## for stability of long sequence
    # vec = scipy.signal.sosfilt(sos, vec)
    f, t, tmp_signal = scipy.signal.stft(vec, fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary='zeros')
    noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1)  # [batch * chn, nf, nt, 2]
    noisy_signal[np.isnan(noisy_signal)] = 0
    noisy_signal[np.isinf(noisy_signal)] = 0
    X_input = normalize_batch(noisy_signal)

    feed = {model.X: X_input, model.drop_rate: 0, model.is_training: False}
    preds = sess.run(model.preds, feed_dict=feed)

    _, denoised_signal = scipy.signal.istft(
        (noisy_signal[..., 0] + noisy_signal[..., 1] * 1j) * preds[..., 0],
        fs=FS,
        nperseg=NPERSEG,
        nfft=NFFT,
        boundary='zeros',
    )
    # _, denoised_noise = scipy.signal.istft(
    #     (noisy_signal[..., 0] + noisy_signal[..., 1] * 1j) * preds[..., 1],
    #     fs=FS,
    #     nperseg=NPERSEG,
    #     nfft=NFFT,
    #     boundary='zeros',
    # )

    denoised_signal = np.reshape(denoised_signal, [nbt, nch, nt])
    denoised_signal = np.transpose(denoised_signal, [0, 2, 1])

    result = meta.copy()
    result.vec = denoised_signal.tolist()
    return result


class Data(BaseModel):
    # id: Union[List[str], str]
    # timestamp: Union[List[str], str]
    # vec: Union[List[List[List[float]]], List[List[float]]]
    id: List[str]
    timestamp: List[str]
    vec: List[List[List[float]]]
    dt: float = 0.01


@app.post("/predict")
def predict(data: Data):

    denoised = get_prediction(data)

    return denoised


@app.get("/healthz")
def healthz():
    return {"status": "ok"}