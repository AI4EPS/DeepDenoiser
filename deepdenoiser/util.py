import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy import signal
from tqdm import tqdm

from data_reader import Config

matplotlib.use('agg')


def plot_result(epoch, num, figure_dir, preds, X, Y, mode="valid"):
    config = Config()
    for i in range(min(num, len(X))):

        t, noisy_signal = scipy.signal.istft(
            X[i, :, :, 0] + X[i, :, :, 1] * 1j, fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros'
        )
        t, ideal_denoised_signal = scipy.signal.istft(
            (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * Y[i, :, :, 0],
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )
        t, denoised_signal = scipy.signal.istft(
            (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0],
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )

        plt.figure(i)
        fig_size = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(fig_size * [1.5, 1.5])
        plt.subplot(4, 2, 1)
        plt.pcolormesh(np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j), vmin=0, vmax=2)
        plt.title("Noisy signal")
        plt.gca().set_xticklabels([])
        plt.subplot(4, 2, 2)
        plt.plot(t, noisy_signal, label='Noisy signal', linewidth=0.1)
        signal_ylim = plt.gca().get_ylim()
        plt.gca().set_xticklabels([])
        plt.legend(loc='lower left')
        plt.margins(x=0)

        plt.subplot(4, 2, 3)
        plt.pcolormesh(Y[i, :, :, 0], vmin=0, vmax=1)
        plt.gca().set_xticklabels([])
        plt.title("Y")
        plt.subplot(4, 2, 4)
        plt.pcolormesh(preds[i, :, :, 0], vmin=0, vmax=1)
        plt.title("$\hat{Y}$")
        plt.gca().set_xticklabels([])

        plt.subplot(4, 2, 5)
        plt.pcolormesh(np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * Y[i, :, :, 0], vmin=0, vmax=2)
        plt.title("Ideal denoised signal")
        plt.gca().set_xticklabels([])
        plt.subplot(4, 2, 6)
        plt.pcolormesh(np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0], vmin=0, vmax=2)
        plt.title("Denoised signal")
        plt.gca().set_xticklabels([])

        plt.subplot(4, 2, 7)
        plt.plot(t, ideal_denoised_signal, label='Ideal denoised signal', linewidth=0.1)
        plt.ylim(signal_ylim)
        plt.xlabel("Time (s)")
        plt.legend(loc='lower left')
        plt.margins(x=0)
        plt.subplot(4, 2, 8)
        plt.plot(t, denoised_signal, label='Denoised signal', linewidth=0.1)
        plt.ylim(signal_ylim)
        plt.xlabel("Time (s)")
        plt.legend(loc='lower left')
        plt.margins(x=0)

        plt.tight_layout()
        plt.gcf().align_labels()
        plt.savefig(os.path.join(figure_dir, "epoch{:03d}_{:03d}_{:}.png".format(epoch, i, mode)), bbox_inches='tight')
        # plt.savefig(os.path.join(figure_dir, "epoch%03d_%03d.pdf" % (epoch, i)), bbox_inches='tight')
        plt.close(i)
    return 0


def plot_result_thread(i, epoch, preds, X, Y, figure_dir, mode="valid"):
    config = Config()
    t, noisy_signal = scipy.signal.istft(
        X[i, :, :, 0] + X[i, :, :, 1] * 1j, fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros'
    )
    t, ideal_denoised_signal = scipy.signal.istft(
        (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * Y[i, :, :, 0],
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=config.nfft,
        boundary='zeros',
    )
    t, denoised_signal = scipy.signal.istft(
        (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0],
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=config.nfft,
        boundary='zeros',
    )

    plt.figure(i)
    fig_size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(fig_size * [1.5, 1.5])
    plt.subplot(4, 2, 1)
    plt.pcolormesh(np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j), vmin=0, vmax=2)
    plt.title("Noisy signal")
    plt.gca().set_xticklabels([])
    plt.subplot(4, 2, 2)
    plt.plot(t, noisy_signal, 'k', label='Noisy signal', linewidth=0.5)
    signal_ylim = plt.gca().get_ylim()
    plt.gca().set_xticklabels([])
    plt.legend(loc='lower left')
    plt.margins(x=0)

    plt.subplot(4, 2, 3)
    plt.pcolormesh(Y[i, :, :, 0], vmin=0, vmax=1)
    plt.gca().set_xticklabels([])
    plt.title("Y")
    plt.subplot(4, 2, 4)
    plt.pcolormesh(preds[i, :, :, 0], vmin=0, vmax=1)
    plt.title("$\hat{Y}$")
    plt.gca().set_xticklabels([])

    plt.subplot(4, 2, 5)
    plt.pcolormesh(np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * Y[i, :, :, 0], vmin=0, vmax=2)
    plt.title("Ideal denoised signal")
    plt.gca().set_xticklabels([])
    plt.subplot(4, 2, 6)
    plt.pcolormesh(np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0], vmin=0, vmax=2)
    plt.title("Denoised signal")
    plt.gca().set_xticklabels([])

    plt.subplot(4, 2, 7)
    plt.plot(t, ideal_denoised_signal, 'k', label='Ideal denoised signal', linewidth=0.5)
    plt.ylim(signal_ylim)
    plt.xlabel("Time (s)")
    plt.legend(loc='lower left')
    plt.margins(x=0)
    plt.subplot(4, 2, 8)
    plt.plot(t, denoised_signal, 'k', label='Denoised signal', linewidth=0.5)
    plt.ylim(signal_ylim)
    plt.xlabel("Time (s)")
    plt.legend(loc='lower left')
    plt.margins(x=0)

    plt.tight_layout()
    plt.gcf().align_labels()
    plt.savefig(os.path.join(figure_dir, "epoch{:03d}_{:03d}_{:}.png".format(epoch, i, mode)), bbox_inches='tight')
    plt.close(i)
    return 0


def postprocessing_test(
    i, preds, X, fname, figure_dir=None, result_dir=None, signal_FT=None, noise_FT=None, data_dir=None
):
    if (figure_dir is not None) or (result_dir is not None):
        config = Config()
        t1, noisy_signal = scipy.signal.istft(
            X[i, :, :, 0] + X[i, :, :, 1] * 1j, fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros'
        )
        t1, denoised_signal = scipy.signal.istft(
            (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0],
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )
        t1, denoised_noise = scipy.signal.istft(
            (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * (1 - preds[i, :, :, 0]),
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )
        t1, signal = scipy.signal.istft(
            signal_FT[i, :, :], fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros'
        )
        t1, noise = scipy.signal.istft(
            noise_FT[i, :, :], fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary='zeros'
        )

    if result_dir is not None:
        try:
            np.savez(
                os.path.join(result_dir, fname[i].decode()),
                preds=preds[i],
                X=X[i],
                signal_FT=signal_FT[i],
                noise_FT=noise_FT[i],
                noisy_signal=noisy_signal,
                denoised_signal=denoised_signal,
                denoised_noise=denoised_noise,
                signal=signal,
                noise=noise,
            )
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(result_dir, fname[i].decode())), exist_ok=True)
            np.savez(
                os.path.join(result_dir, fname[i].decode()),
                preds=preds[i],
                X=X[i],
                signal_FT=signal_FT[i],
                noise_FT=noise_FT[i],
                noisy_signal=noisy_signal,
                denoised_signal=denoised_signal,
                denoised_noise=denoised_noise,
                signal=signal,
                noise=noise,
            )

    if figure_dir is not None:
        t_FT = np.linspace(config.time_range[0], config.time_range[1], X.shape[2])
        f_FT = np.linspace(config.freq_range[0], config.freq_range[1], X.shape[1])

        raw_data = None
        if data_dir is not None:
            raw_data = np.load(os.path.join(data_dir, fname[i].decode().split('/')[-1]))
            itp = raw_data['itp']
            its = raw_data['its']
            ix1 = (750 - 50) / 100
            ix2 = (750 + (its - itp) + 50) / 100
            if ix2 - ix1 > 3:
                ix2 = ix1 + 3

        box = dict(boxstyle='round', facecolor='white', alpha=1)

        text_loc = [0.05, 0.8]
        plt.figure(i)
        fig_size = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(fig_size * [1, 2])
        plt.subplot(511)
        plt.pcolormesh(t_FT, f_FT, np.abs(signal_FT[i, :, :]), vmin=0, vmax=1)
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(i)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )
        plt.subplot(512)
        plt.pcolormesh(t_FT, f_FT, np.abs(noise_FT[i, :, :]), vmin=0, vmax=1)
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(ii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )
        plt.subplot(513)
        plt.pcolormesh(t_FT, f_FT, np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j), vmin=0, vmax=1)
        plt.ylabel("Frequency (Hz)", fontsize='large')
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(iii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )
        plt.subplot(514)
        plt.pcolormesh(t_FT, f_FT, np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0], vmin=0, vmax=1)
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(iv)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )
        plt.subplot(515)
        plt.pcolormesh(t_FT, f_FT, np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 1], vmin=0, vmax=1)
        plt.xlabel("Time (s)", fontsize='large')
        plt.text(
            text_loc[0],
            text_loc[1],
            '(v)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        try:
            plt.savefig(os.path.join(figure_dir, fname[i].decode().rstrip('.npz') + '_FT.png'), bbox_inches='tight')
            # plt.savefig(os.path.join(figure_dir, fname[i].decode().rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
        except FileNotFoundError:
            os.makedirs(
                os.path.dirname(os.path.join(figure_dir, fname[i].decode().rstrip('.npz') + '_FT.png')), exist_ok=True
            )
            plt.savefig(os.path.join(figure_dir, fname[i].decode().rstrip('.npz') + '_FT.png'), bbox_inches='tight')
            # plt.savefig(os.path.join(figure_dir, fname[i].decode().rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
        plt.close(i)

        text_loc = [0.05, 0.8]
        plt.figure(i)
        fig_size = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(fig_size * [1, 2])

        ax3 = plt.subplot(513)
        plt.plot(t1, noisy_signal, 'k', linewidth=0.5, label='Noisy signal')
        plt.legend(loc='lower left', fontsize='medium')
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        plt.ylim([-np.max(np.abs(noisy_signal)), np.max(np.abs(noisy_signal))])
        signal_ylim = [-np.max(np.abs(noisy_signal[100:-100])), np.max(np.abs(noisy_signal[100:-100]))]
        plt.ylim(signal_ylim)
        plt.ylabel("Amplitude", fontsize='large')
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(iii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        ax1 = plt.subplot(511)
        plt.plot(t1, signal, 'k', linewidth=0.5, label='Signal')
        plt.legend(loc='lower left', fontsize='medium')
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        plt.ylim(signal_ylim)
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(i)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        plt.subplot(512)
        plt.plot(t1, noise, 'k', linewidth=0.5, label='Noise')
        plt.legend(loc='lower left', fontsize='medium')
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        plt.ylim([-np.max(np.abs(noise)), np.max(np.abs(noise))])
        noise_ylim = [-np.max(np.abs(noise[100:-100])), np.max(np.abs(noise[100:-100]))]
        plt.ylim(noise_ylim)
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(ii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        ax4 = plt.subplot(514)
        plt.plot(t1, denoised_signal, 'k', linewidth=0.5, label='Recovered signal')
        plt.legend(loc='lower left', fontsize='medium')
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        plt.ylim(signal_ylim)
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(iv)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        plt.subplot(515)
        plt.plot(t1, denoised_noise, 'k', linewidth=0.5, label='Recovered noise')
        plt.legend(loc='lower left', fontsize='medium')
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        plt.xlabel("Time (s)", fontsize='large')
        plt.ylim(noise_ylim)
        plt.text(
            text_loc[0],
            text_loc[1],
            '(v)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        if data_dir is not None:
            axins = inset_axes(
                ax1, width=2.0, height=1.0, loc='upper right', bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes
            )
            axins.plot(t1, signal, 'k', linewidth=0.5)
            x1, x2 = ix1, ix2
            y1 = -np.max(np.abs(signal[(t1 > ix1) & (t1 < ix2)]))
            y2 = -y1
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")

            axins = inset_axes(
                ax3, width=2.0, height=1.0, loc='upper right', bbox_to_anchor=(1, 0.3), bbox_transform=ax3.transAxes
            )
            axins.plot(t1, noisy_signal, 'k', linewidth=0.5)
            x1, x2 = ix1, ix2
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax3, axins, loc1=1, loc2=3, fc="none", ec="0.5")

            axins = inset_axes(
                ax4, width=2.0, height=1.0, loc='upper right', bbox_to_anchor=(1, 0.5), bbox_transform=ax4.transAxes
            )
            axins.plot(t1, denoised_signal, 'k', linewidth=0.5)
            x1, x2 = ix1, ix2
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax4, axins, loc1=1, loc2=3, fc="none", ec="0.5")

        plt.savefig(os.path.join(figure_dir, fname[i].decode().rstrip('.npz') + '_wave.png'), bbox_inches='tight')
        # plt.savefig(os.path.join(figure_dir, fname[i].decode().rstrip('.npz')+'_wave.pdf'), bbox_inches='tight')
        plt.close(i)

    return


def postprocessing_pred(i, preds, X, fname, figure_dir=None, result_dir=None):

    if (result_dir is not None) or (figure_dir is not None):
        config = Config()

        t1, noisy_signal = scipy.signal.istft(
            (X[i, :, :, 0] + X[i, :, :, 1] * 1j),
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )
        t1, denoised_signal = scipy.signal.istft(
            (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0],
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )
        t1, denoised_noise = scipy.signal.istft(
            (X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 1],
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )

    if result_dir is not None:
        try:
            np.savez(
                os.path.join(result_dir, fname[i]),
                noisy_signal=noisy_signal,
                denoised_signal=denoised_signal,
                denoised_noise=denoised_noise,
                t=t1,
            )
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(result_dir, fname[i])))
            np.savez(
                os.path.join(result_dir, fname[i]),
                noisy_signal=noisy_signal,
                denoised_signal=denoised_signal,
                denoised_noise=denoised_noise,
                t=t1,
            )

    if figure_dir is not None:

        t_FT = np.linspace(config.time_range[0], config.time_range[1], X.shape[2])
        f_FT = np.linspace(config.freq_range[0], config.freq_range[1], X.shape[1])

        box = dict(boxstyle='round', facecolor='white', alpha=1)
        text_loc = [0.05, 0.77]

        plt.figure(i)
        fig_size = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(fig_size * [1, 1.2])
        vmax = np.std(np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j)) * 1.8

        plt.subplot(311)
        plt.pcolormesh(
            t_FT,
            f_FT,
            np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j),
            vmin=0,
            vmax=vmax,
            shading='auto',
            label='Noisy signal',
        )
        plt.gca().set_xticklabels([])
        plt.text(
            text_loc[0],
            text_loc[1],
            '(i)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )
        plt.subplot(312)
        plt.pcolormesh(
            t_FT,
            f_FT,
            np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 0],
            vmin=0,
            vmax=vmax,
            shading='auto',
            label='Recovered signal',
        )
        plt.gca().set_xticklabels([])
        plt.ylabel("Frequency (Hz)", fontsize='large')
        plt.text(
            text_loc[0],
            text_loc[1],
            '(ii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )
        plt.subplot(313)
        plt.pcolormesh(
            t_FT,
            f_FT,
            np.abs(X[i, :, :, 0] + X[i, :, :, 1] * 1j) * preds[i, :, :, 1],
            vmin=0,
            vmax=vmax,
            shading='auto',
            label='Recovered noise',
        )
        plt.xlabel("Time (s)", fontsize='large')
        plt.text(
            text_loc[0],
            text_loc[1],
            '(iii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        try:
            plt.savefig(os.path.join(figure_dir, fname[i].rstrip('.npz') + '_FT.png'), bbox_inches='tight')
            # plt.savefig(os.path.join(figure_dir, fname[i].split('/')[-1].rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(figure_dir, fname[i].rstrip('.npz') + '_FT.png')), exist_ok=True)
            plt.savefig(os.path.join(figure_dir, fname[i].rstrip('.npz') + '_FT.png'), bbox_inches='tight')
            # plt.savefig(os.path.join(figure_dir, fname[i].split('/')[-1].rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
        plt.close(i)

        plt.figure(i)
        fig_size = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(fig_size * [1, 1.2])

        ax4 = plt.subplot(311)
        plt.plot(t1, noisy_signal, 'k', label='Noisy signal', linewidth=0.5)
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        signal_ylim = [-np.max(np.abs(noisy_signal[100:-100])), np.max(np.abs(noisy_signal[100:-100]))]
        plt.ylim(signal_ylim)
        plt.gca().set_xticklabels([])
        plt.legend(loc='lower left', fontsize='medium')
        plt.text(
            text_loc[0],
            text_loc[1],
            '(i)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        ax5 = plt.subplot(312)
        plt.plot(t1, denoised_signal, 'k', label='Recovered signal', linewidth=0.5)
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        plt.ylim(signal_ylim)
        plt.gca().set_xticklabels([])
        plt.ylabel("Amplitude", fontsize='large')
        plt.legend(loc='lower left', fontsize='medium')
        plt.text(
            text_loc[0],
            text_loc[1],
            '(ii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        plt.subplot(313)
        plt.plot(t1, denoised_noise, 'k', label='Recovered noise', linewidth=0.5)
        plt.xlim([np.around(t1[0]), np.around(t1[-1])])
        plt.ylim(signal_ylim)
        plt.xlabel("Time (s)", fontsize='large')
        plt.legend(loc='lower left', fontsize='medium')
        plt.text(
            text_loc[0],
            text_loc[1],
            '(iii)',
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize="medium",
            fontweight="bold",
            bbox=box,
        )

        plt.savefig(os.path.join(figure_dir, fname[i].rstrip('.npz') + '_wave.png'), bbox_inches='tight')
        # plt.savefig(os.path.join(figure_dir, fname[i].rstrip('.npz')+'_wave.pdf'), bbox_inches='tight')
        plt.close(i)

    return


def save_results(mask, X, fname, t0, save_signal=True, save_noise=True, result_dir="results"):

    config = Config()

    if save_signal:
        _, denoised_signal = scipy.signal.istft(
            (X[..., 0] + X[..., 1] * 1j) * mask[..., 0],
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )  # nbt, nch, nst, nt
        denoised_signal = np.transpose(denoised_signal, [0, 3, 2, 1])  # nbt, nt, nst, nch,
    if save_noise:
        _, denoised_noise = scipy.signal.istft(
            (X[..., 0] + X[..., 1] * 1j) * mask[..., 1],
            fs=config.fs,
            nperseg=config.nperseg,
            nfft=config.nfft,
            boundary='zeros',
        )
        denoised_noise = np.transpose(denoised_noise, [0, 3, 2, 1])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i in range(len(X)):
        np.savez(
            os.path.join(result_dir, fname[i]),
            data=denoised_signal[i] if save_signal else None,
            noise=denoised_noise[i] if save_noise else None,
            t0=t0[i],
        )


def plot_figures(mask, X, fname, figure_dir="figures"):

    config = Config()

    # plot the last channel
    mask = mask[-1, -1, ...]  # nch, nst, nf, nt, 2 => nf, nt, 2
    X = X[-1, -1, ...]

    t1, noisy_signal = scipy.signal.istft(
        (X[..., 0] + X[..., 1] * 1j),
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=config.nfft,
        boundary='zeros',
    )
    t1, denoised_signal = scipy.signal.istft(
        (X[..., 0] + X[..., 1] * 1j) * mask[..., 0],
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=config.nfft,
        boundary='zeros',
    )
    t1, denoised_noise = scipy.signal.istft(
        (X[..., 0] + X[..., 1] * 1j) * mask[..., 1],
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=config.nfft,
        boundary='zeros',
    )

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    t_FT = np.linspace(config.time_range[0], config.time_range[1], X.shape[1])
    f_FT = np.linspace(config.freq_range[0], config.freq_range[1], X.shape[0])

    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.05, 0.77]

    plt.figure()
    fig_size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(fig_size * [1, 1.2])
    vmax = np.std(np.abs(X[:, :, 0] + X[:, :, 1] * 1j)) * 1.8

    plt.subplot(311)
    plt.pcolormesh(
        t_FT,
        f_FT,
        np.abs(X[:, :, 0] + X[:, :, 1] * 1j),
        vmin=0,
        vmax=vmax,
        shading='auto',
        label='Noisy signal',
    )
    plt.gca().set_xticklabels([])
    plt.text(
        text_loc[0],
        text_loc[1],
        '(i)',
        horizontalalignment='center',
        transform=plt.gca().transAxes,
        fontsize="medium",
        fontweight="bold",
        bbox=box,
    )
    plt.subplot(312)
    plt.pcolormesh(
        t_FT,
        f_FT,
        np.abs(X[:, :, 0] + X[:, :, 1] * 1j) * mask[:, :, 0],
        vmin=0,
        vmax=vmax,
        shading='auto',
        label='Recovered signal',
    )
    plt.gca().set_xticklabels([])
    plt.ylabel("Frequency (Hz)", fontsize='large')
    plt.text(
        text_loc[0],
        text_loc[1],
        '(ii)',
        horizontalalignment='center',
        transform=plt.gca().transAxes,
        fontsize="medium",
        fontweight="bold",
        bbox=box,
    )
    plt.subplot(313)
    plt.pcolormesh(
        t_FT,
        f_FT,
        np.abs(X[:, :, 0] + X[:, :, 1] * 1j) * mask[:, :, 1],
        vmin=0,
        vmax=vmax,
        shading='auto',
        label='Recovered noise',
    )
    plt.xlabel("Time (s)", fontsize='large')
    plt.text(
        text_loc[0],
        text_loc[1],
        '(iii)',
        horizontalalignment='center',
        transform=plt.gca().transAxes,
        fontsize="medium",
        fontweight="bold",
        bbox=box,
    )

    try:
        plt.savefig(os.path.join(figure_dir, fname.rstrip('.npz') + '_FT.png'), bbox_inches='tight')
        # plt.savefig(os.path.join(figure_dir, fname[i].split('/')[-1].rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
    except FileNotFoundError:
        os.makedirs(os.path.dirname(os.path.join(figure_dir, fname.rstrip('.npz') + '_FT.png')), exist_ok=True)
        plt.savefig(os.path.join(figure_dir, fname.rstrip('.npz') + '_FT.png'), bbox_inches='tight')
        # plt.savefig(os.path.join(figure_dir, fname[i].split('/')[-1].rstrip('.npz')+'_FT.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure()
    fig_size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(fig_size * [1, 1.2])

    ax4 = plt.subplot(311)
    plt.plot(t1, noisy_signal, 'k', label='Noisy signal', linewidth=0.5)
    plt.xlim([np.around(t1[0]), np.around(t1[-1])])
    signal_ylim = [-np.max(np.abs(noisy_signal)), np.max(np.abs(noisy_signal))]
    if signal_ylim[0] != signal_ylim[1]:
        plt.ylim(signal_ylim)
    plt.gca().set_xticklabels([])
    plt.legend(loc='lower left', fontsize='medium')
    plt.text(
        text_loc[0],
        text_loc[1],
        '(i)',
        horizontalalignment='center',
        transform=plt.gca().transAxes,
        fontsize="medium",
        fontweight="bold",
        bbox=box,
    )

    ax5 = plt.subplot(312)
    plt.plot(t1, denoised_signal, 'k', label='Recovered signal', linewidth=0.5)
    plt.xlim([np.around(t1[0]), np.around(t1[-1])])
    if signal_ylim[0] != signal_ylim[1]:
        plt.ylim(signal_ylim)
    plt.gca().set_xticklabels([])
    plt.ylabel("Amplitude", fontsize='large')
    plt.legend(loc='lower left', fontsize='medium')
    plt.text(
        text_loc[0],
        text_loc[1],
        '(ii)',
        horizontalalignment='center',
        transform=plt.gca().transAxes,
        fontsize="medium",
        fontweight="bold",
        bbox=box,
    )

    plt.subplot(313)
    plt.plot(t1, denoised_noise, 'k', label='Recovered noise', linewidth=0.5)
    plt.xlim([np.around(t1[0]), np.around(t1[-1])])
    if signal_ylim[0] != signal_ylim[1]:
        plt.ylim(signal_ylim)
    plt.xlabel("Time (s)", fontsize='large')
    plt.legend(loc='lower left', fontsize='medium')
    plt.text(
        text_loc[0],
        text_loc[1],
        '(iii)',
        horizontalalignment='center',
        transform=plt.gca().transAxes,
        fontsize="medium",
        fontweight="bold",
        bbox=box,
    )

    plt.savefig(os.path.join(figure_dir, fname.rstrip('.npz') + '_wave.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(figure_dir, fname[i].rstrip('.npz')+'_wave.pdf'), bbox_inches='tight')
    plt.close()

    return


if __name__ == "__main__":
    pass
