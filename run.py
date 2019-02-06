import numpy as np
import tensorflow as tf
import argparse
import os
import time
import logging
from unet import UNet
from data_reader import *
from util import *
from tqdm import tqdm
import multiprocessing
from functools import partial


def read_flags():
  """Returns flags"""

  parser = argparse.ArgumentParser()

  parser.add_argument("--mode",
                      default="train",
                      help="train/valid/test/debug")

  parser.add_argument("--epochs",
                      default=100,
                      type=int,
                      help="Number of epochs (default: 10)")

  parser.add_argument("--batch_size",
                      default=200,
                      type=int,
                      help="Batch size")

  parser.add_argument("--learning_rate",
                      default=0.001,
                      type=float,
                      help="learning rate")

  parser.add_argument("--decay_step",
                      default=-1,
                      type=int,
                      help="decay step")

  parser.add_argument("--decay_rate",
                      default=0.95,
                      type=float,
                      help="decay rate")

  parser.add_argument("--momentum",
                      default=0.9,
                      type=float,
                      help="momentum")

  parser.add_argument("--filters_root",
                      default=8,
                      type=int,
                      help="filters root")

  parser.add_argument("--depth",
                      default=6,
                      type=int,
                      help="depth")

  parser.add_argument("--kernel_size",
                      nargs="+",
                      type=int,
                      default=[3, 3],
                      help="kernel size")

  parser.add_argument("--pool_size",
                      nargs="+",
                      type=int,
                      default=[2, 2],
                      help="pool size")

  parser.add_argument("--drop_rate",
                      default=0,
                      type=float,
                      help="drop out rate")

  parser.add_argument("--dilation_rate",
                      nargs="+",
                      type=int,
                      default=[1, 1],
                      help="dilation_rate")

  parser.add_argument("--loss_type",
                      default="cross_entropy",
                      help="loss type: cross_entropy, IOU, mean_squared")

  parser.add_argument("--weight_decay",
                      default=0,
                      type=float,
                      help="weight decay")

  parser.add_argument("--optimizer",
                      default="adam",
                      help="optimizer: adam, momentum")

  parser.add_argument("--summary",
                      default=True,
                      type=bool,
                      help="summary")

  parser.add_argument("--class_weights",
                      nargs="+",
                      default=[1, 1],
                      type=float,
                      help="class weights")

  parser.add_argument("--logdir",
                      default="log",
                      help="Tensorboard log directory (default: log)")

  parser.add_argument("--ckdir",
                      default=None,
                      help="Checkpoint directory (default: None)")

  parser.add_argument("--plot_number",
                      default=10,
                      type=int,
                      help="plotting trainning result")


  parser.add_argument("--fpred",
                      default="preds.npz",
                      help="ouput file name of test data")
  parser.add_argument("--plot_pred",
                      default=True,
                      type=bool,
                      help="if plot prediction results")
  parser.add_argument("--save_pred",
                      default=True,
                      type=bool,
                      help="if save npz file of prediction")

  flags = parser.parse_args()
  return flags


def set_config(flags, data_reader):
  config = Config()

  config.X_shape = data_reader.X_shape
  config.n_channel = config.X_shape[-1]
  config.Y_shape = data_reader.Y_shape
  config.n_class = config.Y_shape[-1]

  config.depths = flags.depth
  config.filters_root = flags.filters_root
  config.kernel_size = flags.kernel_size
  config.pool_size = flags.pool_size
  config.dilation_rate = flags.dilation_rate
  config.batch_size = flags.batch_size
  config.class_weights = flags.class_weights
  config.loss_type = flags.loss_type
  config.weight_decay = flags.weight_decay
  config.optimizer = flags.optimizer

  config.learning_rate = flags.learning_rate
  if (flags.decay_step == -1) and (flags.mode == 'train'):
    config.decay_step = data_reader.n_train // flags.batch_size
  else:
    config.decay_step = flags.decay_step
  config.decay_rate = flags.decay_rate
  config.momentum = flags.momentum

  config.summary = flags.summary
  config.drop_rate = flags.drop_rate
  config.class_weights = flags.class_weights

  return config


def train_fn(flags, data_reader):
  current_time = time.strftime("%m%d%H%M%S")
  logging.info("Train : %s" % current_time)
  log_dir = os.path.join(flags.logdir, current_time)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  fig_dir = os.path.join(log_dir, 'figure')
  if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

  config = set_config(flags, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.name_scope('Input_Batch'):
    batch = data_reader.dequeue(flags.batch_size * 3) # 3: three channels

  model = UNet(config)
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)

    if flags.ckdir is not None:
      logging.info("restoring models...")
      latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
      saver.restore(sess, latest_check_point)

    threads = tf.train.start_queue_runners(
        sess=sess, coord=data_reader.coord)
    data_reader.start_threads(sess, n_threads=8)

    flog = open(os.path.join(log_dir, 'loss.log'), 'w')

    total_step = 0
    mean_loss = 0
    for epoch in range(flags.epochs):
      progressbar = tqdm(range(0, data_reader.n_train,
                               flags.batch_size), desc="Epoch {}".format(epoch))
      for step in progressbar:
        X_batch, Y_batch = sess.run(batch)
        loss_batch, preds_batch, logits_batch = model.train_on_batch(
                              sess, X_batch, Y_batch, summary_writer, flags.drop_rate)
        total_step += 1
        mean_loss += (loss_batch-mean_loss)/total_step
        progressbar.set_description("Epoch {}, loss={:.6f}, mean loss={:.6f}".format(epoch, loss_batch, mean_loss))

        flog.write("epoch: {}, step: {}, loss: {}, mean loss: {}\n".format(epoch, step//flags.batch_size, loss_batch, mean_loss))
        flog.flush()

      plot_result(epoch, flags.plot_number, fig_dir,
                   logits_batch, preds_batch,
                   X_batch, Y_batch)
      saver.save(sess, os.path.join(log_dir, "model.ckpt"))
    flog.close()
  return 0


def valid_fn(flags, data_reader, fig_dir=None, save_results=True):
  current_time = time.strftime("%m%d%H%M%S")
  logging.info("{}: {}".format(flags.mode, current_time))
  log_dir = os.path.join(flags.logdir, flags.mode, current_time)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if fig_dir is None:
    fig_dir = log_dir

  config = set_config(flags, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.name_scope('Input_Batch'):
    batch = data_reader.dequeue(flags.batch_size)

  model = UNet(config)
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
    saver.restore(sess, latest_check_point)

    threads = tf.train.start_queue_runners(
        sess=sess, coord=data_reader.coord)
    
    data_reader.start_threads(sess, n_threads=1, mode=flags.mode)    

    if save_results:
      losses = []
      preds = []
      logits = []
      representation = []
      X = []
      Y = []
      signal = []
      noise = []
      noisy_signal = []
      fname = []

    flog = open(os.path.join(log_dir, 'loss.log'), 'w')
    total_step = 0
    mean_loss = 0
    progressbar = tqdm(range(0, data_reader.n_valid-flags.batch_size+1, flags.batch_size), desc=flags.mode)
    for step in progressbar:
      X_batch, Y_batch, ratio_batch, signal_batch, noise_batch, fname_batch = sess.run(batch)
      loss_batch, preds_batch, logits_batch = model.valid_on_batch(
                                              sess, X_batch, Y_batch, summary_writer)
      total_step += 1
      mean_loss += (loss_batch-mean_loss)/total_step
      progressbar.set_description("{}, loss={:.6f}, mean loss={:6f}".format(flags.mode, loss_batch, mean_loss))

      flog.write("step: {}, loss: {}\n".format(step, loss_batch))
      flog.flush()

      with multiprocessing.Pool(multiprocessing.cpu_count()*2) as pool:
        pool.map(partial(plot_thread, 
                         fig_dir=fig_dir,
                         logits=logits_batch, 
                         preds=preds_batch, 
                         X=X_batch*ratio_batch[:,np.newaxis,np.newaxis,np.newaxis], Y=Y_batch, 
                         signal_FT=signal_batch, 
                         noise_FT=noise_batch, 
                         epoch=step//flags.batch_size,
                         fname=fname_batch), 
                        #  fname=fname_batch, data_dir="../Dataset/NPZ_PS/HNE_HNN_HNZ"), 
                 range(len(X_batch)))

      if save_results:
        losses.append(loss_batch)
        preds.append(preds_batch)
        logits.append(logits_batch)
        X.append(X_batch*ratio_batch[:,np.newaxis,np.newaxis,np.newaxis])
        Y.append(Y_batch)
        signal.append(signal_batch)
        noise.append(noise_batch)
        fname.extend(fname_batch)

    flog.close()

    if save_results:
      preds = np.vstack(preds)
      logits = np.vstack(logits)
      X = np.vstack(X)
      Y = np.vstack(Y)
      signal = np.vstack(signal)
      noise = np.vstack(noise)

    if save_results:
      np.savez(os.path.join(log_dir, flags.fpred), preds=preds, logits=logits, X=X, Y=Y, signal=signal, noise=noise, fname=fname)

  return 0


def debug_fn(flags, data_reader, fig_dir=None, save_results=True):
  current_time = time.strftime("%m%d%H%M%S")
  logging.info("Debug: %s" % current_time)
  log_dir = os.path.join(flags.logdir, "debug", current_time)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if fig_dir is None:
    fig_dir = log_dir

  config = set_config(flags, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.name_scope('Input_Batch'):
    batch = data_reader.dequeue(flags.batch_size)

  model = UNet(config)
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
    saver.restore(sess, latest_check_point)

    threads = tf.train.start_queue_runners(
        sess=sess, coord=data_reader.coord)
    data_reader.start_threads(sess, n_threads=1)

    flog = open(os.path.join(log_dir, 'loss.log'), 'w')
    total_step = 0
    mean_loss = 0
    progressbar = tqdm(range(0, data_reader.n_valid-flags.batch_size+1, flags.batch_size), desc="Debug")
    for step in progressbar:
      X_batch, Y_batch = sess.run(batch)
      loss_batch, preds_batch, logits_batch = model.valid_on_batch(
                                              sess, X_batch, Y_batch, summary_writer)
      total_step += 1
      mean_loss += (loss_batch-mean_loss)/total_step
      progressbar.set_description("Debug, loss={:.6f}, mean loss={:.6f}".format(loss_batch, mean_loss))

      flog.write("step: {}, loss: {}\n".format(step, loss_batch))
      flog.flush()
    flog.close()

  return 0

def pred_fn(flags, data_reader, fig_dir=None, npz_dir=None, log_dir=None):
  current_time = time.strftime("%m%d%H%M%S")
  if log_dir is None:
    log_dir = os.path.join(flags.logdir, "pred", current_time)
  logging.info("Pred log: %s" % log_dir)
  # logging.info("Dataset size: {}".format(data_reader.num_data))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if fig_dir is None:
    fig_dir = os.path.join(log_dir, 'figure')
    os.makedirs(fig_dir, exist_ok=True)
  if npz_dir is None:
    npz_dir = os.path.join(log_dir, 'npz')
    os.makedirs(npz_dir, exist_ok=True)

  config = set_config(flags, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.name_scope('Input_Batch'):
    batch = data_reader.dequeue(flags.batch_size)

  model = UNet(config)
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
    saver.restore(sess, latest_check_point)

    threads = tf.train.start_queue_runners(
        sess=sess, coord=data_reader.coord)
    data_reader.start_threads(sess, n_threads=20)

    preds = []
    logits = []
    X = []
    fname = []

    pool = multiprocessing.Pool(multiprocessing.cpu_count()*5)
    for step in tqdm(range(0, data_reader.n_signal, flags.batch_size), desc="Pred"):
      X_batch, ratio_batch, fname_batch = sess.run(batch)
      preds_batch, logits_batch = model.predict_on_batch(sess, X_batch)
      
      preds.append(preds_batch)
      logits.append(logits_batch)
      X.append(X_batch*ratio_batch[:,np.newaxis,np.newaxis,np.newaxis])
      fname.extend(fname_batch)

      # if flags.save_pred:
      #   print('Saving pred')
      #   pool.map(partial(istft_thread, 
      #                   npz_dir=npz_dir,
      #                   logits=logits_batch, 
      #                   preds=preds_batch, 
      #                   X=X_batch*ratio_batch[:,np.newaxis,np.newaxis,np.newaxis],
      #                   epoch=step//flags.batch_size,
      #                   fname=fname_batch), 
      #                 #  fname=fname_batch, data_dir="../Dataset/Demo/HNE_HNN_HNZ"), 
      #             range(len(X_batch)))
      # with multiprocessing.Pool(multiprocessing.cpu_count()*5) as pool:
      if flags.plot_pred:
        print('Ploting pred')
        pool.map(partial(plot_pred_thread, 
                        fig_dir=fig_dir,
                        logits=logits_batch, 
                        preds=preds_batch, 
                        X=X_batch*ratio_batch[:,np.newaxis,np.newaxis,np.newaxis],
                        # epoch=step//flags.batch_size,
                        fname=fname_batch), 
                      #  fname=fname_batch, data_dir="../Dataset/Demo/HNE_HNN_HNZ"), 
                range(len(X_batch)))
        print('Finishing plotting')

      if step + flags.batch_size >= data_reader.n_signal:
        for t in threads:
          t.join()
        sess.run(data_reader.queue.close())
    
    pool.close()
    if flags.save_pred:
      preds = np.vstack(preds)
      logits = np.vstack(logits)
      X = np.vstack(X)
      np.savez(os.path.join(log_dir, flags.fpred), preds=preds, logits=logits, X=X, fname=fname)

  return 0

def main(flags):

  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

  coord = tf.train.Coordinator()

  if flags.mode == "train":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader(
          signal_dir="../Dataset/STFT/HNE_HNN_HNZ",
          signal_list="../Dataset/STFT/HNE_HNN_HNZ.csv",
          noise_dir="../Dataset/STFT_Noise/HNE_HNN_HNZ",
          noise_list="../Dataset/STFT_Noise/HNE_HNN_HNZ.csv",
          queue_size=flags.batch_size*2,
          coord=coord)
    logging.info("Dataset size: training %d, validation %d, test %d" % 
        (data_reader.n_train, data_reader.n_valid, data_reader.n_test))
    train_fn(flags, data_reader)
  
  elif flags.mode == "valid" or flags.mode == "test":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader_valid(
          signal_dir="../Dataset/STFT/HNE_HNN_HNZ",
          signal_list="../Dataset/STFT/HNE_HNN_HNZ.csv",
          noise_dir="../Dataset/STFT_Noise/HNE_HNN_HNZ",
          noise_list="../Dataset/STFT_Noise/HNE_HNN_HNZ.csv",
          queue_size=flags.batch_size*2,
          coord=coord)
    logging.info("Dataset Size: training %d, validation %d, test %d" % 
        (data_reader.n_train, data_reader.n_valid, data_reader.n_test))
    valid_fn(flags, data_reader)

  elif flags.mode == "debug":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader(
          signal_dir="../Dataset/STFT/HNE_HNN_HNZ",
          signal_list="../Dataset/STFT/HNE_HNN_HNZ.csv",
          noise_dir="../Dataset/STFT_Noise/HNE_HNN_HNZ",
          noise_list="../Dataset/STFT_Noise/HNE_HNN_HNZ.csv",
          queue_size=flags.batch_size*2,
          coord=coord)
    logging.info("Dataset Size: training %d, validation %d, test %d" % 
        (data_reader.n_train, data_reader.n_valid, data_reader.n_test))
    debug_fn(flags, data_reader)

  elif flags.mode == "pred":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader_pred(
          signal_dir="../DAS/NPZ",
          signal_list="../DAS/fname.csv",
          queue_size=flags.batch_size*2,
          coord=coord)
    pred_fn(flags, data_reader)

  else:
    print("mode should be: train, valid, test, debug or pred")

  coord.request_stop()
  coord.join()

  return


if __name__ == '__main__':
  flags = read_flags()
  main(flags)
