from __future__ import print_function, division
import numpy as np
import scipy.signal
import pandas as pd
pd.options.mode.chained_assignment = None
import tensorflow as tf
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
  nt = 4100
  X_shape = [31, int(np.ceil(nt/(nperseg//2)))+1, 2]
  Y_shape = [31, int(np.ceil(nt/(nperseg//2)))+1, n_class]
  queue_size = 10
  plot = False
  use_seed = False
  use_buffer = True
  # signal_shape = [31, 201]
  # noise_shape = signal_shape
  # noise_mean = 2
  # noise_std = 1
  # noise_low = 1
  # noise_high = 5
  # snr_threshold = 10

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

  def thread_main(self, sess, n_threads=1, start=0):
    stop = False
    while not stop:
      index = list(range(start, self.n_signal, n_threads))
      np.random.shuffle(index)
      for i in index:
        fname_signal = os.path.join(self.signal_dir, self.signal.iloc[i]['fname'])
        try:
          if fname_signal not in self.buffer_signal:
            data_FT = []
            tmp_data = np.load(fname_signal)['data'][:self.config.nt]
            tmp_data -= np.mean(tmp_data)
            f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
            self.buffer_signal[fname_signal] = {'data_FT': tmp_FT}
          meta_signal = self.buffer_signal[fname_signal]
        except:
          logging.error("Failed reading signal: {}".format(fname_signal))
          continue

        fname_noise = os.path.join(self.noise_dir, self.noise.sample(n=1).iloc[0]['fname'])
        try:
          if fname_noise not in self.buffer_noise:
            data_FT = []
            tmp_data = np.load(fname_noise)['data'][:self.config.nt]
            tmp_data -= np.mean(tmp_data)
            f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
            self.buffer_noise[fname_noise] = {'data_FT': tmp_FT}
          meta_noise = self.buffer_noise[fname_noise]
        except:
          logging.error("Failed reading noise: {}".format(fname_noise))
          continue

        if self.coord.should_stop():
          stop = True
          break

        tmp_noise = meta_noise['data_FT']
        if np.isinf(tmp_noise).any() or np.isnan(tmp_noise).any() or (not np.any(tmp_noise)):
            continue
        tmp_noise = tmp_noise/np.std(tmp_noise)

        # tmp_signal = np.zeros([self.X_shape[0], self.X_shape[1]], dtype=np.complex_)
        # if np.random.random() < 0.9:
          # shift = np.random.randint(-self.X_shape[1], 1, None, 'int')
          # tmp_signal[:, -shift:] = meta_signal['data_FT'][:, self.X_shape[1]:2*self.X_shape[1]+shift, j]
        tmp_signal = meta_signal["data_FT"]
        if np.isinf(tmp_signal).any() or np.isnan(tmp_signal).any() or (not np.any(tmp_signal)):
          continue
        tmp_signal = tmp_signal/np.std(tmp_signal)
        
        if np.random.random() < 0.1:
          tmp_signal = np.fliplr(tmp_signal)

        # ratio = np.random.chisquare(2)
        ratio = np.random.randn() 

        tmp_noisy_signal = (tmp_signal + ratio * tmp_noise)
        noisy_signal = np.stack([tmp_noisy_signal.real, tmp_noisy_signal.imag], axis=-1)
        if np.isnan(noisy_signal).any() or np.isinf(noisy_signal).any():
          continue

        noisy_signal = noisy_signal/np.std(noisy_signal)
        tmp_mask = np.abs(tmp_signal)/(np.abs(tmp_signal) + np.abs(ratio * tmp_noise) + 1e-10)
        tmp_mask[tmp_mask >= 1] = 1
        tmp_mask[tmp_mask <= 0] = 0
        mask = np.zeros([tmp_mask.shape[0], tmp_mask.shape[1], self.n_class])
        mask[:, :, 0] = tmp_mask
        mask[:, :, 1] = 1-tmp_mask

        # return tmp_signal, tmp_noise, noisy_signal, mask
        sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
                                          self.target_placeholder: mask})

  def start_threads(self, sess, n_threads=8):
    for i in range(n_threads):
      thread = threading.Thread(target=self.thread_main, args=(sess, n_threads, i))
      thread.daemon = True
      thread.start()
      self.threads.append(thread)
    return self.threads

class DataReader_valid(DataReader):
  pass

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
               queue_size,
               coord,
               config=Config()):
    self.config = config
    signal_list = pd.read_csv(signal_list)
    self.signal = signal_list
    self.n_signal = len(self.signal)
    self.signal_dir = signal_dir
    self.n_class = config.n_class
    FT_shape = self.get_shape()
    self.X_shape = [FT_shape[0], FT_shape[1], 2]
    self.Y_shape = [FT_shape[0], FT_shape[1], self.n_class]

    #self.n_class = config.n_class
    #self.X_shape = config.X_shape
    #self.Y_shape = config.Y_shape

    #self.coord = coord
    #self.threads = []
    #self.queue_size = queue_size
    #self.add_placeholder()

  def get_shape(self):
    fname = self.signal.iloc[0]['fname']
    data_signal = np.load(os.path.join(self.signal_dir, fname))
    f, t, tmp_signal = scipy.signal.stft(scipy.signal.detrend(np.squeeze(data_signal['data'])),
                                         fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
    return tmp_signal.shape
  
  #def add_placeholder(self):
  #  self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
  #  self.ratio_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
  #  self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
  #  self.queue = tf.PaddingFIFOQueue(self.queue_size,
  #                                   ['float32', 'float32', 'string'],
  #                                   shapes=[self.config.X_shape, [], []])
  #  self.enqueue = self.queue.enqueue([self.sample_placeholder, self.ratio_placeholder, self.fname_placeholder])

  #def dequeue(self, num_elements):
  #  output = self.queue.dequeue_up_to(num_elements)
  #  return output
  def __len__(self):
      return self.n_signal

  #def thread_main(self, sess, n_threads=1, start=0):
  def __getitem__(self, i):
  #  index = list(range(start, self.n_signal, n_threads))
  #  shift = 0
  #  for i in index:
    #for i in range(self.n_signal):
    fname = self.signal.iloc[i]['fname']
    data_signal = np.load(os.path.join(self.signal_dir, fname))
    f, t, tmp_signal = scipy.signal.stft(scipy.signal.detrend(np.squeeze(data_signal['data'])),
                                         fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
    noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1)
    noisy_signal[np.isnan(noisy_signal)] = 0
    noisy_signal[np.isinf(noisy_signal)] = 0
    std_noisy_signal = np.std(noisy_signal)
    if std_noisy_signal != 0:
      noisy_signal = noisy_signal/std_noisy_signal
    #sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
    #                                  self.ratio_placeholder: std_noisy_signal,
    #                                  self.fname_placeholder: fname})
    return noisy_signal, std_noisy_signal, fname


if __name__ == "__main__":
  data_reader = DataReader(
    signal_dir="/Users/weiqiang/Research/BrianZhu/Data/",
    signal_list="/Users/weiqiang/Research/BrianZhu/signal.csv",
    noise_dir="/Users/weiqiang/Research/BrianZhu/Data/",
    noise_list="/Users/weiqiang/Research/BrianZhu/noise.csv",
    queue_size=1,
    coord=None)

  sess = None
  signal, noise, data, mask = data_reader.thread_main(sess, n_threads=1, start=0)

  
