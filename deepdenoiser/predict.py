import argparse
import logging
import multiprocessing
import os
import time
from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_reader import DataReader_pred, normalize_batch
from model import UNet
from util import *

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def read_args():
    """Returns args"""

    parser = argparse.ArgumentParser()

    parser.add_argument("--format", default="numpy", type=str, help="Input data format: numpy or mseed")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs (default: 10)")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
    parser.add_argument("--output_dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--model_dir", default=None, help="Checkpoint directory (default: None)")
    parser.add_argument("--sampling_rate", default=100, type=int, help="sampling rate of pred data")
    parser.add_argument("--data_dir", default="./Dataset/pred/", help="Input file directory")
    parser.add_argument("--data_list", default="./Dataset/pred.csv", help="Input csv file")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure")
    parser.add_argument("--save_signal", action="store_true", help="If save denoised signal")
    parser.add_argument("--save_noise", action="store_true", help="If save denoised noise")

    args = parser.parse_args()
    return args


def pred_fn(args, data_reader, figure_dir=None, result_dir=None, log_dir=None):
    current_time = time.strftime("%y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(args.log_dir, "pred", current_time)
    logging.info("Pred log: %s" % log_dir)
    # logging.info("Dataset size: {}".format(data_reader.num_data))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.plot_figure:
        figure_dir = os.path.join(log_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)
    if args.save_signal or args.save_noise:
        result_dir = os.path.join(log_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)

    with tf.compat.v1.name_scope('Input_Batch'):
        data_batch = data_reader.dataset(args.batch_size)

    # model = UNet(input_batch=data_batch, mode='pred')
    model = UNet(mode='pred')
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # sess_config.log_device_placement = False

    with tf.compat.v1.Session(config=sess_config) as sess:

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        latest_check_point = tf.train.latest_checkpoint(args.model_dir)
        logging.info(f"restoring models: {latest_check_point}")
        saver.restore(sess, latest_check_point)

        if args.plot_figure:
            num_pool = multiprocessing.cpu_count()
        else:
            num_pool = 2
        multiprocessing.set_start_method('spawn')
        pool = multiprocessing.Pool(num_pool)
        for _ in tqdm(range(0, data_reader.n_signal, args.batch_size), desc="Pred"):
            X_batch, fname_batch, t0_batch = sess.run(data_batch)
            nbt, nch, nst, nf, nt, nimg = X_batch.shape
            X_batch_ = np.reshape(X_batch, [nbt * nch * nst, nf, nt, nimg])
            X_batch_ = normalize_batch(X_batch_)
            preds_batch = sess.run(
                model.preds,
                feed_dict={model.X: X_batch_, model.drop_rate: 0, model.is_training: False},
            )
            preds_batch = np.reshape(preds_batch, [nbt, nch, nst, nf, nt, preds_batch.shape[-1]])
            # preds_batch, X_batch, ratio_batch, fname_batch = sess.run(
            #     [model.preds, data_batch[0], data_batch[1], data_batch[2]],
            #     feed_dict={model.drop_rate: 0, model.is_training: False},
            # )

            if args.save_signal or args.save_noise:
                save_results(
                    preds_batch,
                    X_batch,
                    fname=[x.decode() for x in fname_batch],
                    t0=[x.decode() for x in t0_batch],
                    save_signal=args.save_signal,
                    save_noise=args.save_noise,
                    result_dir=result_dir,
                )

            if args.plot_figure:
                pool.starmap(
                    partial(
                        plot_figures,
                        figure_dir=figure_dir,
                    ),
                    zip(preds_batch, X_batch, [x.decode() for x in fname_batch]),
                )

        pool.close()

    return 0


def main(args):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    with tf.compat.v1.name_scope('create_inputs'):
        data_reader = DataReader_pred(
            format=args.format, signal_dir=args.data_dir, signal_list=args.data_list, sampling_rate=args.sampling_rate
        )
    logging.info("Dataset Size: {}".format(data_reader.n_signal))
    pred_fn(args, data_reader, log_dir=args.output_dir)

    return 0


if __name__ == '__main__':
    args = read_args()
    main(args)
