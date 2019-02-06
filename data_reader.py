from __future__ import print_function, division
import numpy as np
import scipy.signal
import pandas as pd
pd.options.mode.chained_assignment = None
import tensorflow as tf
import threading
import random
import os


class Config():
  seed = 100
  n_class = 2
  data_dir = "../"
  num_repeat_noise = 1
  fs = 100
  dt = 1.0/fs
  freq_range = [0, fs/2]
  time_range = [0, 30]
  nperseg = 30
  nfft = 60
  plot = False
  X_shape = [31, 201, 2]
  Y_shape = [31, 201, n_class]
  signal_shape = [31, 201]
  noise_shape = signal_shape
  use_seed = False
  queue_size = 10

class DataReader(object):

  def __init__(self,
               signal_dir,
               signal_list,
               noise_dir,
               noise_list,
               queue_size,
               coord,
               config=Config()):
    self.config = config
    signal_list = pd.read_csv(signal_list, sep='\s+', header=0)
    noise_list = pd.read_csv(noise_list, sep='\s+', header=0)

    self.noise = noise_list
    self.signal = signal_list[signal_list['snr'] > 10]
    self.n_noise = len(self.noise)
    self.n_signal = len(self.signal)
    self.n_class = config.n_class
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.index_train = 0
    self.index_valid = 0
    self.index_test = 0
    self.split()

    self.coord = coord
    self.threads = []
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.queue = tf.PaddingFIFOQueue(queue_size,
                                     ['float32', 'float32'],
                                     shapes=[self.config.X_shape, self.config.Y_shape])
    self.enqueue = self.queue.enqueue(
        [self.sample_placeholder, self.target_placeholder])

  def split(self, ratio=[0.8, 0.1, 0.1]):
    self.signal['station_id'] = self.signal['network'] + \
        '_'+self.signal['station']
    self.noise['station_id'] = self.noise['network'] + \
        '_'+self.noise['station']

    self.train_signal = self.signal.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[0], random_state=self.config.seed, replace=False))
    tmp_signal = self.signal.drop(self.train_signal.index)
    self.valid_signal = tmp_signal.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[1]/(ratio[1]+ratio[2]), random_state=self.config.seed, replace=False))
    self.test_signal = tmp_signal.drop(self.valid_signal.index)

    self.train_noise = self.noise.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[0], random_state=self.config.seed, replace=False))
    tmp_noise = self.noise.drop(self.train_noise.index)
    self.valid_noise = tmp_noise.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[1]/(ratio[1]+ratio[2]), random_state=self.config.seed, replace=False))
    self.test_noise = tmp_noise.drop(self.valid_noise.index)

    self.n_train = len(self.train_signal)
    self.n_valid = len(self.valid_signal)
    self.n_test = len(self.test_signal)

  def dequeue(self, num_elements):
    output = self.queue.dequeue_many(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    stop = False
    while not stop:
      index = list(range(start, self.n_train, n_threads))
      random.shuffle(index)
      for i in index:
        data_signal = np.load(self.config.data_dir +
                              self.train_signal.iloc[i]['fname'])
        data_noise = np.load(self.config.data_dir +
                             self.train_noise.sample(n=1).iloc[0]['fname'])

        if self.coord.should_stop():
          stop = True
          break

        for j in range(3):
          tmp_noise = data_noise['data'][...,
                                         np.random.randint(0, 3, None, 'int')]
          tmp_signal = np.zeros_like(tmp_noise)
          if self.config.use_seed:
            np.random.seed(self.config.seed+i*3+j)
          if np.random.random() < 0.9:
            shift = np.random.randint(-self.X_shape[1], 1, None, 'int')
            tmp_signal[:, -shift:] = data_signal['data'][:,
                                                         self.X_shape[1]:2*self.X_shape[1]+shift, j]
          
          if np.random.random() < 0.1:
            tmp_signal = np.fliplr(tmp_signal)

          if (np.isnan(tmp_signal).any() or np.isnan(tmp_noise).any()
                  or np.isinf(tmp_signal).any() or np.isinf(tmp_noise).any()):
            continue

          for dum in range(self.config.num_repeat_noise):
            if self.config.use_seed:
              np.random.seed(self.config.seed+(i*3+j)*(self.config.num_repeat_noise)+dum)
            ratio = 0
            while ratio <= 0:
              ratio = 2 + np.random.randn()
            tmp_noisy_signal = (ratio * tmp_noise + tmp_signal)
            noisy_signal = np.stack(
                [tmp_noisy_signal.real, tmp_noisy_signal.imag], axis=-1)
            noisy_signal = noisy_signal/np.std(noisy_signal)
            # tmp_mask = (np.abs(tmp_signal)) / (np.abs(tmp_noisy_signal) + 1e-4)
            tmp_mask = np.abs(tmp_signal)/(np.abs(tmp_signal) + np.abs(ratio * tmp_noise) + 1e-4)
            tmp_mask[tmp_mask >= 1] = 1
            tmp_mask[tmp_mask <= 0] = 0
            mask = np.zeros(
                [tmp_mask.shape[0], tmp_mask.shape[1], self.n_class])
            mask[:, :, 0] = tmp_mask
            mask[:, :, 1] = 1-tmp_mask

            sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
                                              self.target_placeholder: mask})

  def start_threads(self, sess, n_threads=8):
    for i in range(n_threads):
      thread = threading.Thread(
          target=self.thread_main, args=(sess, n_threads, i))
      thread.daemon = True  # Thread will close when parent quits.
      thread.start()
      self.threads.append(thread)
    return self.threads

class DataReader_valid(object):

  def __init__(self,
               signal_dir,
               signal_list,
               noise_dir,
               noise_list,
               queue_size,
               coord,
               config=Config()):
    self.config = config
    signal_list = pd.read_csv(signal_list, sep='\s+', header=0)
    noise_list = pd.read_csv(noise_list, sep='\s+', header=0)

    self.noise = noise_list
    self.signal = signal_list[signal_list['snr'] > 10]
    self.n_noise = len(self.noise)
    self.n_signal = len(self.signal)
    self.n_class = config.n_class
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.index_train = 0
    self.index_valid = 0
    self.index_test = 0
    self.split()

    self.coord = coord
    self.threads = []
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.ratio_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.signal_placeholder = tf.placeholder(dtype=tf.complex64, shape=None)
    self.noise_placeholder = tf.placeholder(dtype=tf.complex64, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.queue = tf.PaddingFIFOQueue(queue_size,
                                     ['float32', 'float32', 'float32', 'complex64', 'complex64', 'string'],
                                     shapes=[self.config.X_shape, self.config.Y_shape, [], self.config.signal_shape, self.config.noise_shape, []])
    self.enqueue = self.queue.enqueue(
        [self.sample_placeholder, self.target_placeholder, self.ratio_placeholder, self.signal_placeholder, self.noise_placeholder, self.fname_placeholder])

  def split(self, ratio=[0.8, 0.1, 0.1]):
    self.signal['station_id'] = self.signal['network'] + \
        '_'+self.signal['station']
    self.noise['station_id'] = self.noise['network'] + \
        '_'+self.noise['station']

    self.train_signal = self.signal.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[0], random_state=self.config.seed, replace=False))
    tmp_signal = self.signal.drop(self.train_signal.index)
    self.valid_signal = tmp_signal.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[1]/(ratio[1]+ratio[2]), random_state=self.config.seed, replace=False))
    self.test_signal = tmp_signal.drop(self.valid_signal.index)

    self.train_noise = self.noise.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[0], random_state=self.config.seed, replace=False))
    tmp_noise = self.noise.drop(self.train_noise.index)
    self.valid_noise = tmp_noise.groupby('station_id', group_keys=False).apply(
        lambda x: x.sample(frac=ratio[1]/(ratio[1]+ratio[2]), random_state=self.config.seed, replace=False))
    self.test_noise = tmp_noise.drop(self.valid_noise.index)

    self.n_train = len(self.train_signal)
    self.n_valid = len(self.valid_signal)
    self.n_test = len(self.test_signal)

  def dequeue(self, num_elements):
    output = self.queue.dequeue_many(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0, mode='valid'):
    if mode == 'valid':
      index = list(range(start, self.n_valid, n_threads))
    elif mode == 'test':
      index = list(range(start, self.n_test, n_threads))
    for i in index:
      if mode == 'valid':
        data_signal = np.load(self.config.data_dir +
                              self.valid_signal.iloc[i]['fname'])
        data_noise = np.load(self.config.data_dir +
                             self.valid_noise.sample(n=1,random_state=self.config.seed+i).iloc[0]['fname'])
        fname = self.valid_signal.iloc[i]['fname']
      elif mode == 'test':
        data_signal = np.load(self.config.data_dir +
                              self.test_signal.iloc[i]['fname'])
        data_noise = np.load(self.config.data_dir +
                             self.test_noise.sample(n=1, random_state=self.config.seed+i).iloc[0]['fname'])
        fname = self.test_signal.iloc[i]['fname']

      for j in [2]:
        tmp_noise = data_noise['data'][..., j]
        tmp_signal = np.zeros_like(tmp_noise)
        np.random.seed(self.config.seed + i)
        if True:
        # if False:
          shift = -int(self.X_shape[1]*(1/4))
          if self.X_shape[1]+shift <= 1 or shift > 0:
            shift = 0
          tmp_signal[:, -shift:] = data_signal['data'][:,
                                                       self.X_shape[1]:2*self.X_shape[1]+shift, j]

        if (np.isnan(tmp_signal).any() or np.isnan(tmp_noise).any()
                or np.isinf(tmp_signal).any() or np.isinf(tmp_noise).any()):
          continue

        for dum in [0]:
          ratio = 0
          tmp = 0
          np.random.seed(self.config.seed+i+tmp)
          ratio = 2
          tmp_noisy_signal = (ratio * tmp_noise + tmp_signal)
          noisy_signal = np.stack(
              [tmp_noisy_signal.real, tmp_noisy_signal.imag], axis=-1)
          std_noisy_signal = np.std(noisy_signal)
          noisy_signal = noisy_signal/std_noisy_signal
          tmp_mask = (np.abs(tmp_signal)) / (np.abs(tmp_noisy_signal) + 1e-4)
          tmp_mask[tmp_mask >= 1] = 1
          tmp_mask[tmp_mask <= 0] = 0
          mask = np.zeros(
              [tmp_mask.shape[0], tmp_mask.shape[1], self.n_class])
          mask[:, :, 0] = tmp_mask
          mask[:, :, 1] = 1-tmp_mask
          sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
                                            self.target_placeholder: mask, 
                                            self.ratio_placeholder: std_noisy_signal,
                                            self.signal_placeholder: tmp_signal, 
                                            self.noise_placeholder: ratio*tmp_noise,
                                            self.fname_placeholder: fname})

  def start_threads(self, sess, n_threads=1, mode='valid'):
    for i in range(n_threads):
      thread = threading.Thread(
          target=self.thread_main, args=(sess, n_threads, i, mode))
      thread.daemon = True  # Thread will close when parent quits.
      thread.start()
      self.threads.append(thread)
    return self.threads

class DataReader_pred(object):

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
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.ratio_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.queue = tf.PaddingFIFOQueue(queue_size,
                                     ['float32', 'float32', 'string'],
                                     shapes=[self.config.X_shape, [], []])
    self.enqueue = self.queue.enqueue(
        [self.sample_placeholder, self.ratio_placeholder, self.fname_placeholder])


  def dequeue(self, num_elements):
    output = self.queue.dequeue_many(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.n_signal, n_threads))
    for i in index:

      data_signal = np.load(os.path.join(self.signal_dir,
                            self.signal.iloc[i]['fname'].split('/')[-1]))
      fname = self.signal.iloc[i]['fname']
      if len(data_signal['data'].shape) == 1:
        f, t, tmp_signal = scipy.signal.stft(scipy.signal.detrend(data_signal['data']), fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
      else:
        f, t, tmp_signal = scipy.signal.stft(scipy.signal.detrend(data_signal['data'][..., -1]), fs=self.config.fs, nperseg=self.config.nperseg, nfft=self.config.nfft, boundary='zeros')
      noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1)
      if np.isnan(noisy_signal).any() or np.isinf(noisy_signal).any():
        continue

      std_noisy_signal = np.std(noisy_signal)
      noisy_signal = noisy_signal/std_noisy_signal
      sess.run(self.enqueue, feed_dict={self.sample_placeholder: noisy_signal, 
                                        self.ratio_placeholder: std_noisy_signal,
                                        self.fname_placeholder: fname})

  def start_threads(self, sess, n_threads=1):
    for i in range(n_threads):
      thread = threading.Thread(
          target=self.thread_main, args=(sess, n_threads, i))
      thread.daemon = True  # Thread will close when parent quits.
      thread.start()
      self.threads.append(thread)
    return self.threads


if __name__ == "__main__":
  pass
