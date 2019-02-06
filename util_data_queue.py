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


class DataReader_pred(object):

  def __init__(self,
               signal_dir,
               signal_list,
               queue_size,
               coord,
               config):
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
