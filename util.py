import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from scipy import signal
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from data_reader import Config

def istft_thread(i, npz_dir, logits, preds, X, epoch=0, fname=None):
  config = Config()

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

def plot_result(epoch, num, fig_dir, logits, preds, X, Y):
  config = Config()
  for i in range(num):
    plt.clf()
    plt.subplot(3, 2, 1)
    plt.pcolormesh(np.abs(X[i, :, :, 0]+X[i, :, :, 1]*1j), vmin=0, vmax=1)
    plt.subplot(3, 2, 2)
    plt.pcolormesh(Y[i, :, :, 0], vmin=0, vmax=1)
    plt.subplot(3, 2, 3)
    plt.pcolormesh(np.abs(X[i, :, :, 0]+X[i, :, :, 1]*1j)
                   * preds[i, :, :, 0], vmin=0, vmax=1)
    plt.subplot(3, 2, 4)
    plt.pcolormesh(preds[i, :, :, 0], vmin=0, vmax=1)
    t, origin = scipy.signal.istft(X[i, :, :, 0]+X[i, :, :, 1]*1j, fs=config.fs,
                                   nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
    t, denoised = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0],
                                     fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
    plt.subplot(3, 2, 5)
    plt.plot(t, origin, label='Noisy signal', linewidth=0.1)
    signal_ylim = plt.gca().get_ylim()
    plt.xlabel("Time (s)")
    plt.ylabel('Amplitude')
    plt.subplot(3, 2, 6)
    plt.plot(t, denoised, label='Denoised signal', linewidth=0.1)
    plt.ylim(signal_ylim)
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.gcf().align_labels()
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d.pdf" % (epoch, i)), bbox_inches='tight')


def plot_pred(epoch, num, fig_dir, logits, preds, X):
  config = Config()
  if num > X.shape[0]:
    num = X.shape[0]
  for i in range(num):
    plt.clf()
    plt.subplot(511)
    plt.pcolormesh(np.abs(X[i, :, :, 0]+X[i, :, :, 1]*1j), vmin=0, vmax=1)
    plt.subplot(512)
    plt.pcolormesh(preds[i, :, :, 0], vmin=0, vmax=1)
    plt.subplot(513)
    plt.pcolormesh(np.abs(X[i, :, :, 0]+X[i, :, :, 1]*1j)
                   * preds[i, :, :, 0], vmin=0, vmax=1)
    t, origin = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j), fs=config.fs,
                                   nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
    t, denoised = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0],
                                     fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
    plt.subplot(514)
    plt.plot(t, origin, label='Noisy signal', linewidth=0.1)
    signal_ylim = plt.gca().get_ylim()
    plt.xlabel("Time (s)")
    plt.subplot(515)
    plt.plot(t, denoised, label='Denoised signal', linewidth=0.1)
    plt.ylim(signal_ylim)
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.gcf().align_labels()
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d.pdf" % (epoch, i)), bbox_inches='tight')


def plot_pred_thread(i, fig_dir, logits, preds, X, epoch=0, fname=None, data_dir=None):
  config = Config()
  t_FT = np.linspace(config.time_range[0], config.time_range[1], X.shape[2])
  f_FT = np.linspace(config.freq_range[0], config.freq_range[1], X.shape[1])

  box = dict(boxstyle='round', facecolor='white', alpha=1)
  text_loc = [0.05, 0.77]

  raw_data = None
  if (data_dir is not None) and (fname is not None):
    raw_data = np.load(os.path.join(
        data_dir, fname[i].decode().split('/')[-1]))
    itp = raw_data['itp']
    its = raw_data['its']
    ix1 = (750 - 100)/100
    ix2 = (750 + (its - itp) + 50)/100
    if ix2 - ix1 > 3:
      ix2 = ix1 + 3

  plt.figure(i)
  fig_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(fig_size*[1, 1.2])

  vmax = np.std(np.abs(X[i, :, :, 0]+X[i, :, :, 1]*1j)) * 2

  plt.subplot(311)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j), vmin=0, vmax=vmax, label='Noisy signal')
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)
  plt.subplot(312)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0], vmin=0, vmax=vmax, label='Recovered signal')
  plt.gca().set_xticklabels([])
  plt.ylabel("Frequency (Hz)", fontsize='large')
  plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  plt.subplot(313)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 1], vmin=0, vmax=vmax, label='Recovered noise')
  plt.gca().set_xticklabels([])
  plt.xlabel("Time (s)", fontsize='large')
  plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  if fname is None:
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_FT.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_FT.pdf" % (epoch, i)), bbox_inches='tight')
  else:
    plt.savefig(os.path.join(
        fig_dir, fname[i].decode().split('/')[-1].rstrip('.npz')+'_FT.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(
        # fig_dir, fname[i].decode().split('/')[-1].rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
  plt.close(i)

  plt.figure(i)
  fig_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(fig_size*[1, 1.2])

  t1, origin = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j), fs=config.fs,
                                  nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  t1, denoised = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0],
                                    fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  t1, noise = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 1],
                                    fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')

  ax4 = plt.subplot(311)
  plt.plot(t1, origin, 'k', label='Noisy signal', linewidth=0.5)
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  signal_ylim = [-np.max(np.abs(origin[100:-100])),
                 np.max(np.abs(origin[100:-100]))]
  plt.ylim(signal_ylim)
  plt.gca().set_xticklabels([])
  plt.legend(loc='lower left', fontsize='medium')
  plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  ax5 = plt.subplot(312)
  plt.plot(t1, denoised, 'k', label='Recovered signal', linewidth=0.5)
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim(signal_ylim)
  plt.gca().set_xticklabels([])
  plt.ylabel("Amplitude",  fontsize='large')
  plt.legend(loc='lower left', fontsize='medium')
  plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  plt.subplot(313)
  plt.plot(t1, noise, 'k', label='Recovered noise', linewidth=0.5)
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim(signal_ylim)
  plt.xlabel("Time (s)", fontsize='large')
  plt.legend(loc='lower left', fontsize='medium')
  plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  if (data_dir is not None) and (fname is not None):
    axins = inset_axes(ax4, width=2, height=1, loc='upper right',
                       bbox_to_anchor=(1, 0.4),
                       bbox_transform=ax4.transAxes)
    axins.plot(t1, origin, 'k', linewidth=0.5)
    x1, x2 = ix1, ix2
    y1 = -np.max(np.abs(origin[(t1 > ix1) & (t1 < ix2)]))
    y2 = -y1
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax4, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    axins = inset_axes(ax5, width=2, height=1, loc='upper right',
                       bbox_to_anchor=(1, 0.4),
                       bbox_transform=ax5.transAxes)
    axins.plot(t1, denoised, 'k', linewidth=0.5)
    x1, x2 = ix1, ix2
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax5, axins, loc1=1, loc2=3, fc="none", ec="0.5")

  if fname is None:
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_wave.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_wave.pdf" % (epoch, i)), bbox_inches='tight')
  else:
    plt.savefig(os.path.join(
        fig_dir, fname[i].decode().split('/')[-1].rstrip('.npz')+'_wave.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(
        # fig_dir, fname[i].decode().split('/')[-1].rstrip('.npz')+'_wave.pdf'), bbox_inches='tight')

  plt.close(i)
  print(i)
  return 0


def plot_thread(i, fig_dir, logits, preds, X, Y, signal_FT=None, noise_FT=None, epoch=0, fname=None, data_dir=None):
  config = Config()
  t_FT = np.linspace(config.time_range[0], config.time_range[1], X.shape[2])
  f_FT = np.linspace(config.freq_range[0], config.freq_range[1], X.shape[1])

  raw_data = None
  if (data_dir is not None) and (fname is not None):
    raw_data = np.load(os.path.join(
        data_dir, fname[i].decode().split('/')[-1]))
    itp = raw_data['itp']
    its = raw_data['its']
    ix1 = (750 - 50)/100
    ix2 = (750 + (its - itp) + 50)/100
    if ix2 - ix1 > 3:
      ix2 = ix1 + 3

  box = dict(boxstyle='round', facecolor='white', alpha=1)

  text_loc = [0.05, 0.8]
  plt.figure(i)
  fig_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(fig_size*[1, 2])
  plt.subplot(511)
  plt.pcolormesh(t_FT, f_FT, np.abs(signal_FT[i, :, :]), vmin=0, vmax=1)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)
  plt.subplot(512)
  plt.pcolormesh(t_FT, f_FT, np.abs(noise_FT[i, :, :]), vmin=0, vmax=1)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)
  plt.subplot(513)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j), vmin=0, vmax=1)
  plt.ylabel("Frequency (Hz)", fontsize='large')
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)
  plt.subplot(514)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0], vmin=0, vmax=1)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)
  plt.subplot(515)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 1], vmin=0, vmax=1)
  plt.xlabel("Time (s)", fontsize='large')
  plt.text(text_loc[0], text_loc[1], '(v)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  if fname is None:
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_FT.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_FT.pdf" % (epoch, i)), bbox_inches='tight')
  else:
    plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        '/')[-1].rstrip('.npz')+'_FT.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        # '/')[-1].rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
  plt.close(i)

  text_loc = [0.05, 0.8]
  plt.figure(i)
  fig_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(fig_size*[1, 2])

  ax3 = plt.subplot(513)
  t1, noisy_signal = scipy.signal.istft(
      X[i, :, :, 0]+X[i, :, :, 1]*1j, fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, noisy_signal, 'k', linewidth=0.5, label='Noisy signal')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim([-np.max(np.abs(noisy_signal)), np.max(np.abs(noisy_signal))])
  signal_ylim = [-np.max(np.abs(noisy_signal[100:-100])),
                 np.max(np.abs(noisy_signal[100:-100]))]
  plt.ylim(signal_ylim)
  plt.ylabel("Amplitude", fontsize='large')
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  ax1 = plt.subplot(511)
  t1, signal = scipy.signal.istft(
      signal_FT[i, :, :], fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, signal, 'k', linewidth=0.5, label='Signal')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim(signal_ylim)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  plt.subplot(512)
  t1, noise = scipy.signal.istft(
      noise_FT[i, :, :], fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, noise, 'k', linewidth=0.5, label='Noise')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim([-np.max(np.abs(noise)), np.max(np.abs(noise))])
  noise_ylim = [-np.max(np.abs(noise[100:-100])),
                np.max(np.abs(noise[100:-100]))]
  plt.ylim(noise_ylim)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  ax4 = plt.subplot(514)
  t1, denoised_signal = scipy.signal.istft(
      (X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0], fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, denoised_signal, 'k', linewidth=0.5, label='Recovered signal')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim(signal_ylim)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  plt.subplot(515)
  t1, denoised_noise = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*(
      1-preds[i, :, :, 0]), fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, denoised_noise, 'k', linewidth=0.5, label='Recovered noise')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.xlabel("Time (s)", fontsize='large')
  plt.ylim(noise_ylim)
  plt.text(text_loc[0], text_loc[1], '(v)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  if (data_dir is not None) and (fname is not None):
    axins = inset_axes(ax1, width=2.0, height=1., loc='upper right',
                       bbox_to_anchor=(1, 0.5),
                       bbox_transform=ax1.transAxes)
    axins.plot(t1, signal, 'k', linewidth=0.5)
    x1, x2 = ix1, ix2
    y1 = -np.max(np.abs(signal[(t1 > ix1) & (t1 < ix2)]))
    y2 = -y1
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    axins = inset_axes(ax3, width=2.0, height=1., loc='upper right',
                       bbox_to_anchor=(1, 0.3),
                       bbox_transform=ax3.transAxes)
    axins.plot(t1, noisy_signal, 'k', linewidth=0.5)
    x1, x2 = ix1, ix2
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax3, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    axins = inset_axes(ax4, width=2.0, height=1., loc='upper right',
                       bbox_to_anchor=(1, 0.5),
                       bbox_transform=ax4.transAxes)
    axins.plot(t1, denoised_signal, 'k', linewidth=0.5)
    x1, x2 = ix1, ix2
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax4, axins, loc1=1, loc2=3, fc="none", ec="0.5")

  if fname is None:
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_wave.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_wave.pdf" % (epoch, i)), bbox_inches='tight')
  else:
    plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        '/')[-1].rstrip('.npz')+'_wave.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        # '/')[-1].rstrip('.npz')+'_wave.pdf'), bbox_inches='tight')
  plt.close(i)

def plot_thread_nosignal(i, fig_dir, logits, preds, X, Y, signal_FT=None, noise_FT=None, epoch=0, fname=None, data_dir=None):
  config = Config()
  t_FT = np.linspace(config.time_range[0], config.time_range[1], X.shape[2])
  f_FT = np.linspace(config.freq_range[0], config.freq_range[1], X.shape[1])

  raw_data = None
  if (data_dir is not None) and (fname is not None):
    raw_data = np.load(os.path.join(
        data_dir, fname[i].decode().split('/')[-1]))
    itp = raw_data['itp']
    its = raw_data['its']
    ix1 = (750 - 50)/100
    ix2 = (750 + (its - itp) + 50)/100
    if ix2 - ix1 > 3:
      ix2 = ix1 + 3

  box = dict(boxstyle='round', facecolor='white', alpha=1)

  text_loc = [0.05, 0.8]
  plt.figure(i)
  fig_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(fig_size*[1, 1.2])
  plt.subplot(311)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j), vmin=0, vmax=1)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)
  plt.subplot(312)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0], vmin=0, vmax=1)
  plt.ylabel("Frequency (Hz)", fontsize='large')
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)
  plt.subplot(313)
  plt.pcolormesh(t_FT, f_FT, np.abs(
      X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 1], vmin=0, vmax=1)
  plt.xlabel("Time (s)", fontsize='large')
  plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  if fname is None:
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_FT.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_FT.pdf" % (epoch, i)), bbox_inches='tight')
  else:
    plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        '/')[-1].rstrip('.npz')+'_FT.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        # '/')[-1].rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
  plt.close(i)

  text_loc = [0.05, 0.8]
  plt.figure(i)
  fig_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(fig_size*[1, 1.2])

  ax4 = plt.subplot(311)
  t1, noisy_signal = scipy.signal.istft(
      X[i, :, :, 0]+X[i, :, :, 1]*1j, fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, noisy_signal, 'k', linewidth=0.5, label='Noise')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim([-np.max(np.abs(noisy_signal)), np.max(np.abs(noisy_signal))])
  signal_ylim = [-np.max(np.abs(noisy_signal[100:-100])),
                 np.max(np.abs(noisy_signal[100:-100]))]
  plt.ylim(signal_ylim)
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='left',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  t1, signal = scipy.signal.istft(
      signal_FT[i, :, :], fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')

  ax5 = plt.subplot(312)
  t1, denoised_signal = scipy.signal.istft(
      (X[i, :, :, 0]+X[i, :, :, 1]*1j)*preds[i, :, :, 0], fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, denoised_signal, 'k', linewidth=0.5, label='Recovered signal')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.ylim(signal_ylim)
  plt.ylabel("Amplitude", fontsize='large')
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='left',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  plt.subplot(313)
  t1, denoised_noise = scipy.signal.istft((X[i, :, :, 0]+X[i, :, :, 1]*1j)*(
      preds[i, :, :, 1]), fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros')
  plt.plot(t1, denoised_noise, 'k', linewidth=0.5, label='Recovered noise')
  plt.legend(loc='lower left', fontsize='medium')
  plt.xlim([np.around(t1[0]), np.around(t1[-1])])
  plt.xlabel("Time (s)", fontsize='large')
  plt.ylim(signal_ylim)
  plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='left',
           transform=plt.gca().transAxes, fontsize="medium", fontweight="bold", bbox=box)

  if (data_dir is not None) and (fname is not None):
    axins = inset_axes(ax4, width=2.0, height=1., loc='upper right',
                       bbox_to_anchor=(1, 0.4),
                       bbox_transform=ax4.transAxes)
    axins.plot(t1, signal, 'k', linewidth=0.5)
    x1, x2 = ix1, ix2
    y1 = -np.max(np.abs(signal[(t1 > ix1) & (t1 < ix2)]))
    y2 = -y1
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax4, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    axins = inset_axes(ax5, width=2.0, height=1., loc='upper right',
                       bbox_to_anchor=(1, 0.4),
                       bbox_transform=ax5.transAxes)
    axins.plot(t1, noisy_signal, 'k', linewidth=0.5)
    x1, x2 = ix1, ix2
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax5, axins, loc1=1, loc2=3, fc="none", ec="0.5")

  if fname is None:
    plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_wave.png" % (epoch, i)), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, "epoch%03d_%03d_wave.pdf" % (epoch, i)), bbox_inches='tight')
  else:
    plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        '/')[-1].rstrip('.npz')+'_wave.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(fig_dir, fname[i].decode().split(
        # '/')[-1].rstrip('.npz')+'_wave.pdf'), bbox_inches='tight')
  plt.close(i)


if __name__ == "__main__":
    pass

