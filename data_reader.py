from __future__ import print_function, division
import numpy as np
import scipy.signal
import pandas as pd
from tensorflow.python.ops.linalg_ops import norm
pd.options.mode.chained_assignment = None
import tensorflow as tf
from tensorflow.python.util import nest
from scipy.interpolate import interp1d
import threading
import os
import logging


class Config():
  seed = 100
  n_class = 2
  fs = 100
  dt = 1.0/fs
  freq_range = [0, fs/2]
  time_range = [0, 30]
  nperseg = 30
  nfft = 60
  plot = False
  nt = 3000
  X_shape = [31, 201, 2]
  Y_shape = [31, 201, n_class]
  signal_shape = [31, 201]
  noise_shape = signal_shape
  use_seed = False
  queue_size = 10
  noise_mean = 2
  noise_std = 1
  # noise_low = 1
  # noise_high = 5
  use_buffer = True
  snr_threshold = 10

# %%
# def normalize(data, window=3000):
#     """
#     data: nsta, chn, nt
#     """
#     shift = window//2
#     nt = len(data)

#     ## std in slide windows
#     data_pad = np.pad(data, ((window//2, window//2)), mode="reflect")
#     t = np.arange(0, nt, shift, dtype="int")
#     # print(f"nt = {nt}, nt+window//2 = {nt+window//2}")
#     std = np.zeros(len(t))
#     mean = np.zeros(len(t))
#     for i in range(len(std)):
#         std[i] = np.std(data_pad[i*shift:i*shift+window])
#         mean[i] = np.mean(data_pad[i*shift:i*shift+window])

#     t = np.append(t, nt)
#     std = np.append(std, [np.std(data_pad[-window:])])
#     mean = np.append(mean, [np.mean(data_pad[-window:])])

#     # print(t)
#     ## normalize data with interplated std 
#     t_interp = np.arange(nt, dtype="int")
#     std_interp = interp1d(t, std, kind="slinear")(t_interp)
#     mean_interp = interp1d(t, mean, kind="slinear")(t_interp)
#     data = (data - mean_interp)/(std_interp)
#     return data, std_interp

# %%
def normalize(data, window=200):
    """
    data: nsta, chn, nt
    """
    shift = window//2
    nt = data.shape[1]

    ## std in slide windows
    data_pad = np.pad(data, ((0,0), (window//2, window//2), (0,0)), mode="reflect")
    t = np.arange(0, nt, shift, dtype="int")
    # print(f"nt = {nt}, nt+window//2 = {nt+window//2}")
    std = np.zeros(len(t))
    mean = np.zeros(len(t))
    for i in range(len(std)):
        std[i] = np.std(data_pad[:,i*shift:i*shift+window, :])
        mean[i] = np.mean(data_pad[:, i*shift:i*shift+window, :])

    t = np.append(t, nt)
    std = np.append(std, [np.std(data_pad[:, -window:, :])])
    mean = np.append(mean, [np.mean(data_pad[:, -window:, :])])

    # print(t)
    ## normalize data with interplated std 
    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, kind="slinear")(t_interp)
    mean_interp = interp1d(t, mean, kind="slinear")(t_interp)
    data = (data - mean_interp[np.newaxis, :, np.newaxis])/std_interp[np.newaxis, :, np.newaxis]
    return data, std_interp

# %%
def py_func_decorator(output_types=None, output_shapes=None, name=None):
  def decorator(func):
    def call(*args, **kwargs):
      nonlocal output_shapes
      flat_output_types = nest.flatten(output_types)
      # flat_output_types = tf.nest.flatten(output_types)
      flat_values = tf.py_func(
      # flat_values = tf.numpy_function(
        func, 
        inp=args, 
        Tout=flat_output_types,
        name=name
      )
      if output_shapes is not None:
        for v, s in zip(flat_values, output_shapes):
          v.set_shape(s)
      return nest.pack_sequence_as(output_types, flat_values)
      # return tf.nest.pack_sequence_as(output_types, flat_values)
    return call
  return decorator

def dataset_map(iterator, output_types, output_shapes=None, num_parallel_calls=None, name=None):
  dataset = tf.data.Dataset.range(len(iterator))
  @py_func_decorator(output_types, output_shapes, name=name)
  def index_to_entry(idx):
    return iterator[idx]    
  return dataset.map(index_to_entry, num_parallel_calls=num_parallel_calls)

class DataReader(object):

  def __init__(self,
               signal_dir = None,
               signal_list = None,
               noise_dir = None,
               noise_list = None,
               queue_size = None,
               coord = None,
               config=Config()):

    self.config = config

    signal_list = pd.read_csv(signal_list, header=0)
    noise_list = pd.read_csv(noise_list, header=0)
    
    self.signal = signal_list
    self.noise = noise_list
    self.n_signal = len(self.signal)

    self.signal_dir = signal_dir
    self.noise_dir = noise_dir

    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.n_class = config.n_class

    self.coord = coord
    self.threads = []
    self.queue_size = queue_size

    self.add_queue()
    self.buffer_signal = {}
    self.buffer_noise = {}
    self.buffer_channels_signal = {}
    self.buffer_channels_noise = {}

  def add_queue(self):
    with tf.device('/cpu:0'):
      self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
      self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
      self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                      ['float32', 'float32'],
                                      shapes=[self.config.X_shape, self.config.Y_shape])
      self.enqueue = self.queue.enqueue([self.sample_placeholder, self.target_placeholder])
    return 0

  def dequeue(self, num_elements):
    output = self.queue.dequeue_many(num_elements)
    return output

  def get_snr(self, data, itp, dit=300):
    tmp_std = np.std(data[itp-dit:itp])
    if tmp_std > 0:
      return np.std(data[itp:itp+dit])/tmp_std
    else:
      return 0

  def add_event(self, sample, channels, j):
    while np.random.uniform(0, 1) < 0.2:
      shift = None
      if channels not in self.buffer_channels_signal:
        self.buffer_channels_signal[channels] = self.signal[self.signal['channels']==channels]
      fname = os.path.join(self.signal_dir, self.buffer_channels_signal[channels].sample(n=1).iloc[0]['fname'])
      try:
        if fname not in self.buffer_signal:
          meta = np.load(fname)
          data_FT = []
          snr = []
          for i in range(3):
            tmp_data = meta['data'][:, i]
            tmp_itp = meta['itp']
            snr.append(self.get_snr(tmp_data, tmp_itp))
            tmp_data -= np.mean(tmp_data)
            f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
            data_FT.append(tmp_FT)
          data_FT = np.stack(data_FT, axis=-1)
          self.buffer_signal[fname] = {'data_FT': data_FT, 'itp': tmp_itp, 'channels': meta['channels'], 'snr': snr}
        meta_signal = self.buffer_signal[fname]
      except:
        logging.error("Failed reading signal: {}".format(fname))
        continue 
      if meta_signal['snr'][j] > self.config.snr_threshold:
        tmp_signal = np.zeros([self.X_shape[0], self.X_shape[1]], dtype=np.complex_)
        shift = np.random.randint(-self.X_shape[1], 1, None, 'int')
        tmp_signal[:, -shift:] = meta_signal['data_FT'][:, self.X_shape[1]:2*self.X_shape[1]+shift, j]
        if np.isinf(tmp_signal).any() or np.isnan(tmp_signal).any() or (not np.any(tmp_signal)):
          continue
        tmp_signal = tmp_signal/np.std(tmp_signal)
        sample += tmp_signal / np.random.uniform(1, 5)
    return sample

  def thread_main(self, sess, n_threads=1, start=0):
    stop = False
    while not stop:
      index = list(range(start, self.n_signal, n_threads))
      np.random.shuffle(index)
      for i in index:
        fname_signal = os.path.join(self.signal_dir, self.signal.iloc[i]['fname'])
        try:
          if fname_signal not in self.buffer_signal:
            meta = np.load(fname_signal)
            data_FT = []
            snr = []
            for j in range(3):
              tmp_data = meta['data'][..., j]
              tmp_itp = meta['itp']
              snr.append(self.get_snr(tmp_data, tmp_itp))
              tmp_data -= np.mean(tmp_data)
              f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
              data_FT.append(tmp_FT)
            data_FT = np.stack(data_FT, axis=-1)
            self.buffer_signal[fname_signal] = {'data_FT': data_FT, 'itp':tmp_itp, 'channels': meta['channels'], 'snr': snr}
          meta_signal = self.buffer_signal[fname_signal]
        except:
          logging.error("Failed reading signal: {}".format(fname_signal))
          continue
        channels = meta_signal['channels'].tolist()
        start_tp = meta_signal['itp'].tolist()

        if channels not in self.buffer_channels_noise:
          self.buffer_channels_noise[channels] = self.noise[self.noise['channels']==channels]
        fname_noise = os.path.join(self.noise_dir, self.buffer_channels_noise[channels].sample(n=1).iloc[0]['fname'])
        try:
          if fname_noise not in self.buffer_noise:
            meta = np.load(fname_noise)
            data_FT = []
            for i in range(3):
              tmp_data = meta['data'][:self.config.nt, i]
              tmp_data -= np.mean(tmp_data)
              f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
              data_FT.append(tmp_FT)
            data_FT = np.stack(data_FT, axis=-1)
            self.buffer_noise[fname_noise] = {'data_FT': data_FT, 'channels': meta['channels']}
          meta_noise = self.buffer_noise[fname_noise]
        except:
          logging.error("Failed reading noise: {}".format(fname_noise))
          continue

        if self.coord.should_stop():
          stop = True
          break

        j = np.random.choice([0,1,2])
        if meta_signal['snr'][j] <= self.config.snr_threshold:
          continue

        tmp_noise = meta_noise['data_FT'][..., j]
        if np.isinf(tmp_noise).any() or np.isnan(tmp_noise).any() or (not np.any(tmp_noise)):
            continue
        tmp_noise = tmp_noise/np.std(tmp_noise)

        tmp_signal = np.zeros([self.X_shape[0], self.X_shape[1]], dtype=np.complex_)
        if np.random.random() < 0.9:
          shift = np.random.randint(-self.X_shape[1], 1, None, 'int')
          tmp_signal[:, -shift:] = meta_signal['data_FT'][:, self.X_shape[1]:2*self.X_shape[1]+shift, j]
          if np.isinf(tmp_signal).any() or np.isnan(tmp_signal).any() or (not np.any(tmp_signal)):
            continue
          tmp_signal = tmp_signal/np.std(tmp_signal)
          tmp_signal = self.add_event(tmp_signal, channels, j)
        
          if np.random.random() < 0.2:
            tmp_signal = np.fliplr(tmp_signal)

        ratio = 0
        while ratio <= 0:
          ratio = (self.config.noise_mean + np.random.randn() * self.config.noise_std) 
        # ratio = np.random.uniform(self.config.noise_low, self.config.noise_high)
        tmp_noisy_signal = (tmp_signal + ratio * tmp_noise)
        noisy_signal = np.stack([tmp_noisy_signal.real, tmp_noisy_signal.imag], axis=-1)
        if np.isnan(noisy_signal).any() or np.isinf(noisy_signal).any():
          continue
        noisy_signal = noisy_signal/np.std(noisy_signal)
        tmp_mask = np.abs(tmp_signal)/(np.abs(tmp_signal) + np.abs(ratio * tmp_noise) + 1e-4)
        tmp_mask[tmp_mask >= 1] = 1
        tmp_mask[tmp_mask <= 0] = 0
        mask = np.zeros([tmp_mask.shape[0], tmp_mask.shape[1], self.n_class])
        mask[:, :, 0] = tmp_mask
        mask[:, :, 1] = 1-tmp_mask
        sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
                                          self.target_placeholder: mask})

  def start_threads(self, sess, n_threads=8):
    for i in range(n_threads):
      thread = threading.Thread(target=self.thread_main, args=(sess, n_threads, i))
      thread.daemon = True
      thread.start()
      self.threads.append(thread)
    return self.threads

class DataReader_test(DataReader):

  def __init__(self,
               signal_dir = None,
               signal_list = None,
               noise_dir = None,
               noise_list = None,
               queue_size = None,
               coord = None,
               config=Config()):
    self.config = config

    signal_list = pd.read_csv(signal_list, header=0)
    noise_list = pd.read_csv(noise_list, header=0)
    self.signal = signal_list
    self.noise = noise_list
    self.n_signal = len(self.signal)
    
    self.signal_dir = signal_dir
    self.noise_dir = noise_dir

    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.n_class = config.n_class

    self.coord = coord
    self.threads = []
    self.queue_size = queue_size

    self.add_queue()
    self.buffer_signal = {}
    self.buffer_noise = {}
    self.buffer_channels_signal = {}
    self.buffer_channels_noise = {}

  def add_queue(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.ratio_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.signal_placeholder = tf.placeholder(dtype=tf.complex64, shape=None)
    self.noise_placeholder = tf.placeholder(dtype=tf.complex64, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                     ['float32', 'float32', 'float32', 'complex64', 'complex64', 'string'],
                                     shapes=[self.config.X_shape, self.config.Y_shape, [], self.config.signal_shape, self.config.noise_shape, []])
    self.enqueue = self.queue.enqueue(
        [self.sample_placeholder, self.target_placeholder, self.ratio_placeholder, self.signal_placeholder, self.noise_placeholder, self.fname_placeholder])
    return 0

  def dequeue(self, num_elements):
    output = self.queue.dequeue_up_to(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.n_signal, n_threads))
    for i in index:
      np.random.seed(i)

      fname = self.signal.iloc[i]['fname']
      fname_signal = os.path.join(self.signal_dir, fname)
      meta = np.load(fname_signal)
      data_FT = []
      snr = []
      for j in range(3):
        tmp_data = meta['data'][..., j]
        tmp_itp = meta['itp']
        snr.append(self.get_snr(tmp_data, tmp_itp))
        tmp_data -= np.mean(tmp_data)
        f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
        data_FT.append(tmp_FT)
      data_FT = np.stack(data_FT, axis=-1)
      meta_signal = {'data_FT': data_FT, 'itp': tmp_itp, 'channels': meta['channels'], 'snr': snr}
      channels = meta['channels'].tolist()
      start_tp = meta['itp'].tolist()

      if channels not in self.buffer_channels_noise:
        self.buffer_channels_noise[channels] = self.noise[self.noise['channels']==channels]
      fname_noise = os.path.join(self.noise_dir, self.buffer_channels_noise[channels].sample(n=1, random_state=i).iloc[0]['fname'])
      meta = np.load(fname_noise)
      data_FT = []
      for i in range(3):
        tmp_data = meta['data'][:self.config.nt, i]
        tmp_data -= np.mean(tmp_data)
        f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
        data_FT.append(tmp_FT)
      data_FT = np.stack(data_FT, axis=-1)
      meta_noise = {'data_FT': data_FT, 'channels': meta['channels']}

      if self.coord.should_stop():
        stop = True
        break

      j = np.random.choice([0,1,2])
      tmp_noise = meta_noise['data_FT'][..., j]
      if np.isinf(tmp_noise).any() or np.isnan(tmp_noise).any() or (not np.any(tmp_noise)):
          continue
      tmp_noise = tmp_noise/np.std(tmp_noise)

      tmp_signal = np.zeros([self.X_shape[0], self.X_shape[1]], dtype=np.complex_)
      if np.random.random() < 0.9:
        shift = np.random.randint(-self.X_shape[1], 1, None, 'int')
        tmp_signal[:, -shift:] = meta_signal['data_FT'][:, self.X_shape[1]:2*self.X_shape[1]+shift, j]
        if np.isinf(tmp_signal).any() or np.isnan(tmp_signal).any() or (not np.any(tmp_signal)):
          continue
        tmp_signal = tmp_signal/np.std(tmp_signal)
        # tmp_signal = self.add_event(tmp_signal, channels, j)
        # if np.random.random() < 0.2:
        #   tmp_signal = np.fliplr(tmp_signal)

      ratio = 0
      while ratio <= 0:
        ratio = self.config.noise_mean + np.random.randn() * self.config.noise_std 
      tmp_noisy_signal = (tmp_signal + ratio*tmp_noise)
      noisy_signal = np.stack([tmp_noisy_signal.real, tmp_noisy_signal.imag], axis=-1)
      if np.isnan(noisy_signal).any() or np.isinf(noisy_signal).any():
        continue
      std_noisy_signal = np.std(noisy_signal)
      noisy_signal = noisy_signal/std_noisy_signal
      tmp_mask = np.abs(tmp_signal)/(np.abs(tmp_signal) + np.abs(ratio * tmp_noise) + 1e-4)
      tmp_mask[tmp_mask >= 1] = 1
      tmp_mask[tmp_mask <= 0] = 0
      mask = np.zeros([tmp_mask.shape[0], tmp_mask.shape[1], self.n_class])
      mask[:, :, 0] = tmp_mask
      mask[:, :, 1] = 1-tmp_mask

      sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
                                        self.target_placeholder: mask, 
                                        self.ratio_placeholder: std_noisy_signal,
                                        self.signal_placeholder: tmp_signal, 
                                        self.noise_placeholder: ratio*tmp_noise,
                                        self.fname_placeholder: fname})

class DataReader_pred_queue(DataReader):

  def __init__(self,
               signal_dir,
               signal_list,
               queue_size,
               coord,
               config=Config()):
    self.config = config
    signal_list = pd.read_csv(signal_list)
    self.signal = signal_list
    self.n_signal = len(self.signal)
    self.n_class = config.n_class
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.signal_dir = signal_dir

    self.coord = coord
    self.threads = []
    self.queue_size = queue_size
    self.add_placeholder()
  
  def add_placeholder(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.ratio_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                     ['float32', 'float32', 'string'],
                                     shapes=[self.config.X_shape, [], []])
    self.enqueue = self.queue.enqueue([self.sample_placeholder, self.ratio_placeholder, self.fname_placeholder])

  def dequeue(self, num_elements):
    output = self.queue.dequeue_up_to(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.n_signal, n_threads))
    shift = 0
    for i in index:
      fname = self.signal.iloc[i]['fname']
      data_signal = np.load(os.path.join(self.signal_dir, fname))
      f, t, tmp_signal = scipy.signal.stft(scipy.signal.detrend(np.squeeze(data_signal['data'][shift:self.config.nt+shift])),
                                           fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
      noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1)
      if np.isnan(noisy_signal).any() or np.isinf(noisy_signal).any() or (not np.any(noisy_signal)):
        continue
      std_noisy_signal = np.std(noisy_signal)
      if std_noisy_signal == 0:
        continue
      noisy_signal = noisy_signal/std_noisy_signal
      sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
                                        self.ratio_placeholder: std_noisy_signal,
                                        self.fname_placeholder: fname})

class DataReader_pred():

  def __init__(self,
               signal_dir,
               signal_list,
               sampling_rate=100,
               config=Config()):
    self.config = config
    signal_list = pd.read_csv(signal_list)
    self.signal = signal_list
    self.n_signal = len(self.signal)
    self.signal_dir = signal_dir
    self.sampling_rate = sampling_rate
    self.n_class = config.n_class
    FT_shape = self.get_shape()
    self.X_shape = [FT_shape[0], FT_shape[1], 2]
    self.Y_shape = [FT_shape[0], FT_shape[1], self.n_class]
    self.dtype = "float32"

  def get_shape(self):
    fname = self.signal.iloc[0]['fname']
    data = np.load(os.path.join(self.signal_dir, fname))["data"]
    if self.sampling_rate != 100:
      t = np.linspace(0, 1, len(data))
      t_interp = np.linspace(0, 1, np.int(np.around(len(data)*100.0/self.sampling_rate)))
      data = interp1d(t, data, kind="slinear")(t_interp)
    f, t, tmp_signal = scipy.signal.stft(data,
                                         fs=self.config.fs, nperseg=self.config.nperseg, 
                                         nfft=self.config.nfft, boundary='zeros')
    return tmp_signal.shape

  def __len__(self):
      return self.n_signal

  def __getitem__(self, i):
    fname = self.signal.iloc[i]['fname']
    data = np.load(os.path.join(self.signal_dir, fname))["data"]
    if self.sampling_rate != 100:
      t = np.linspace(0, 1, len(data))
      t_interp = np.linspace(0, 1, np.int(np.around(len(data)*100.0/self.sampling_rate)))
      data = interp1d(t, data, kind="slinear")(t_interp)
    f, t, tmp_signal = scipy.signal.stft(data,
                                         fs=self.config.fs, nperseg=self.config.nperseg, 
                                         nfft=self.config.nfft, boundary='zeros')
    noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1)
    noisy_signal[np.isnan(noisy_signal)] = 0
    noisy_signal[np.isinf(noisy_signal)] = 0
    noisy_signal, std_noisy_signal = normalize(noisy_signal)
    return noisy_signal.astype(self.dtype), std_noisy_signal.astype(self.dtype), fname

  def dataset(self, batch_size, num_parallel_calls=4):
    dataset = dataset_map(self, output_types=(self.dtype, self.dtype, "string"),
                          output_shapes=(self.X_shape, self.X_shape[1], 1), 
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size).prefetch(batch_size*3).make_one_shot_iterator().get_next()
    return dataset

if __name__ == "__main__":
  
  # %%
  data_reader = DataReader_pred(signal_dir="./Dataset/yixiao/", 
                                signal_list="./Dataset/yixiao.csv")
  noisy_signal, std_noisy_signal, fname  = data_reader[0]
  print(noisy_signal.shape, std_noisy_signal.shape, fname)
  batch = data_reader.dataset(10)
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  print(sess.run(batch))
