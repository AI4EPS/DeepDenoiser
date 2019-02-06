import tensorflow as tf
import util_data_queue as util_data
import scipy.signal
import numpy as np
import os

def crop_and_concat(net1, net2):
  """
  the size(net1) <= size(net2)
  """
  net1_shape = net1.get_shape().as_list()
  net2_shape = net2.get_shape().as_list()
  # print(net1_shape)
  # print(net2_shape)
  # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
  size = [-1, net1_shape[1], net1_shape[2], -1]
  net2_resize = tf.slice(net2, offsets, size)
  return tf.concat([net1, net2_resize], 3)
  # else:
  #     offsets = [0, (net1_shape[1] - net2_shape[1]) // 2, (net1_shape[2] - net2_shape[2]) // 2, 0]
  #     size = [-1, net2_shape[1], net2_shape[2], -1]
  #     net1_resize = tf.slice(net1, offsets, size)
  #     return tf.concat([net1_resize, net2], 3)

def crop_only(net1, net2):
  """
  the size(net1) <= size(net2)
  """
  net1_shape = net1.get_shape().as_list()
  net2_shape = net2.get_shape().as_list()
  # print(net1_shape)
  # print(net2_shape)
  # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
  size = [-1, net1_shape[1], net1_shape[2], -1]
  net2_resize = tf.slice(net2, offsets, size)
  #return tf.concat([net1, net2_resize], 3)
  return net2_resize


def istft_thread(i, npz_dir, logits, preds, X, epoch=0, fname=None):
  config = util_data.Config()

  t1, origin = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j), fs=config.fs,
                                  nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  t1, denoised = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0],
                                    fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  t1, noise = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 1],
                                    fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')

  if fname is None:
    np.savez(os.path.join(npz_dir, "epoch%03d_%03d.npz" % (epoch, i)), origin=origin, denoised=denoised, noise=noise, t=t1)
  else:
    np.savez(os.path.join(npz_dir, fname[i].decode()), origin=origin, denoised=denoised, noise=noise, t=t1)

  return 0