#import warnings
#warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import os
import time
import logging
from model import UNet
from data_reader import *
from util import *
from tqdm import tqdm
import multiprocessing
from functools import partial


def read_args():
  """Returns args"""

  parser = argparse.ArgumentParser()

  parser.add_argument("--mode",
                      default="train",
                      help="train/valid/test/debug (default: train)")

  parser.add_argument("--epochs",
                      default=10,
                      type=int,
                      help="Number of epochs (default: 10)")

  parser.add_argument("--batch_size",
                      default=20,
                      type=int,
                      help="Batch size (default: 20)")

  parser.add_argument("--learning_rate",
                      default=0.001,
                      type=float,
                      help="learning rate (default: 0.001)")

  parser.add_argument("--decay_step",
                      default=-1,
                      type=int,
                      help="decay step (default: -1)")

  parser.add_argument("--decay_rate",
                      default=0.9,
                      type=float,
                      help="decay rate (default: 0.9)")

  parser.add_argument("--momentum",
                      default=0.9,
                      type=float,
                      help="momentum (default: 0.9)")

  parser.add_argument("--filters_root",
                      default=8,
                      type=int,
                      help="filters root (default: 8)")

  parser.add_argument("--depth",
                      default=6,
                      type=int,
                      help="depth (default: 6)")

  parser.add_argument("--kernel_size",
                      nargs="+",
                      type=int,
                      default=[3, 3],
                      help="kernel size (default: [3, 3]")

  parser.add_argument("--pool_size",
                      nargs="+",
                      type=int,
                      default=[2, 2],
                      help="pool size (default: [2, 2]")

  parser.add_argument("--drop_rate",
                      default=0,
                      type=float,
                      help="drop out rate (default: 0)")

  parser.add_argument("--dilation_rate",
                      nargs="+",
                      type=int,
                      default=[1, 1],
                      help="dilation_rate (default: [1, 1]")

  parser.add_argument("--loss_type",
                      default="cross_entropy",
                      help="loss type: cross_entropy, IOU, mean_squared (default: cross_entropy)")

  parser.add_argument("--weight_decay",
                      default=0,
                      type=float,
                      help="weight decay (default: 0)")

  parser.add_argument("--optimizer",
                      default="adam",
                      help="optimizer: adam, momentum (default: adam)")

  parser.add_argument("--summary",
                      default=True,
                      type=bool,
                      help="summary (default: True)")

  parser.add_argument("--class_weights",
                      nargs="+",
                      default=[1, 1],
                      type=float,
                      help="class weights (default: [1, 1]")

  parser.add_argument("--log_dir",
                      default="log",
                      help="Tensorboard log directory (default: log)")

  parser.add_argument("--model_dir",
                      default=None,
                      help="Checkpoint directory")

  parser.add_argument("--num_plots",
                      default=10,
                      type=int,
                      help="plotting trainning result (default: 10)")

  parser.add_argument("--input_length",
                      default=None,
                      type=int,
                      help="input length")
  parser.add_argument("--sampling_rate",
                      default=100,
                      type=int,
                      help="sampling rate of pred data in Hz (default: 100)")

  parser.add_argument("--train_signal_dir",
                      default="./Dataset/train/",
                      help="Input file directory (default: ./Dataset/train/)")
  parser.add_argument("--train_signal_list",
                      default="./Dataset/train.csv",
                      help="Input csv file (default: ./Dataset/train.csv)")
  parser.add_argument("--train_noise_dir",
                      default="./Dataset/train/",
                      help="Input file directory (default: ./Dataset/train/)")
  parser.add_argument("--train_noise_list",
                      default="./Dataset/train.csv",
                      help="Input csv file (default: ./Dataset/train.csv)")

  parser.add_argument("--valid_signal_dir",
                      default="./Dataset/",
                      help="Input file directory (default: ./Dataset/)")
  parser.add_argument("--valid_signal_list",
                      default=None,
                      help="Input csv file")
  parser.add_argument("--valid_noise_dir",
                      default="./Dataset/",
                      help="Input file directory (default: ./Dataset/)")
  parser.add_argument("--valid_noise_list",
                      default=None,
                      help="Input csv file")

  parser.add_argument("--data_dir",
                      default="./Dataset/pred/",
                      help="Input file directory (default: ./Dataset/pred/)")
  parser.add_argument("--data_list",
                      default="./Dataset/pred.csv",
                      help="Input csv file (default: ./Dataset/pred.csv)")

  parser.add_argument("--output_dir",
                      default=None,
                      help="Output directory")

  parser.add_argument("--fpred",
                      default="preds.npz",
                      help="ouput file name of test data")
  parser.add_argument("--plot_figure",
                      action="store_true",
                      help="If plot figure for test")
  parser.add_argument("--save_result",
                      action="store_true",
                      help="If save result for test")

  args = parser.parse_args()
  return args


def set_config(args, data_reader):
  config = Config()

  config.X_shape = data_reader.X_shape
  config.n_channel = config.X_shape[-1]
  config.Y_shape = data_reader.Y_shape
  config.n_class = config.Y_shape[-1]

  config.depths = args.depth
  config.filters_root = args.filters_root
  config.kernel_size = args.kernel_size
  config.pool_size = args.pool_size
  config.dilation_rate = args.dilation_rate
  config.batch_size = args.batch_size
  config.class_weights = args.class_weights
  config.loss_type = args.loss_type
  config.weight_decay = args.weight_decay
  config.optimizer = args.optimizer

  config.learning_rate = args.learning_rate
  if (args.decay_step == -1) and (args.mode == 'train'):
    config.decay_step = data_reader.n_signal // args.batch_size
  else:
    config.decay_step = args.decay_step
  config.decay_rate = args.decay_rate
  config.momentum = args.momentum

  config.summary = args.summary
  config.drop_rate = args.drop_rate
  config.class_weights = args.class_weights

  return config


def train_fn(args, data_reader, data_reader_valid=None):
  current_time = time.strftime("%y%m%d-%H%M%S")
  log_dir = os.path.join(args.log_dir, current_time)
  logging.info("Training log: {}".format(log_dir))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  figure_dir = os.path.join(log_dir, 'figures')
  if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

  config = set_config(args, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.compat.v1.name_scope('Input_Batch'):
    batch = data_reader.dequeue(args.batch_size)
    if data_reader_valid is not None:
      batch_valid = data_reader_valid.dequeue(args.batch_size)

  model = UNet(config)
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.compat.v1.Session(config=sess_config) as sess:

    summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    if args.model_dir is not None:
      logging.info("restoring models...")
      latest_check_point = tf.train.latest_checkpoint(args.model_dir)
      saver.restore(sess, latest_check_point)
      model.reset_learning_rate(sess, learning_rate=0.01, global_step=0)


    threads = data_reader.start_threads(sess, n_threads=multiprocessing.cpu_count())
    if data_reader_valid is not None:
      threads_valid = data_reader_valid.start_threads(sess, n_threads=multiprocessing.cpu_count())
    flog = open(os.path.join(log_dir, 'loss.log'), 'w')

    total_step = 0
    mean_loss = 0
    pool = multiprocessing.Pool(2)
    for epoch in range(args.epochs):
      progressbar = tqdm(range(0, data_reader.n_signal, args.batch_size), desc="{}: ".format(log_dir.split("/")[-1]))
      for step in progressbar:
        X_batch, Y_batch = sess.run(batch)
        loss_batch = model.train_on_batch(sess, X_batch, Y_batch, summary_writer, args.drop_rate)
        if epoch < 1:
          mean_loss = loss_batch
        else:
          total_step += 1
          mean_loss += (loss_batch-mean_loss)/total_step
        progressbar.set_description("{}: epoch={}, loss={:.6f}, mean loss={:.6f}".format(log_dir.split("/")[-1], epoch, loss_batch, mean_loss))
        flog.write("Epoch: {}, step: {}, loss: {}, mean loss: {}\n".format(epoch, step//args.batch_size, loss_batch, mean_loss))
      saver.save(sess, os.path.join(log_dir, "model_{}.ckpt".format(epoch)))

      ## valid
      if data_reader_valid is not None:
        mean_loss_valid = 0
        total_step_valid = 0
        progressbar = tqdm(range(0, data_reader_valid.n_signal, args.batch_size), desc="Valid: ")
        for step in progressbar:
          X_batch, Y_batch = sess.run(batch_valid)
          loss_batch, preds_batch = model.valid_on_batch(sess, X_batch, Y_batch, summary_writer, args.drop_rate)
          total_step_valid += 1
          mean_loss_valid += (loss_batch-mean_loss_valid)/total_step_valid
          progressbar.set_description("Valid: loss={:.6f}, mean loss={:.6f}".format(loss_batch, mean_loss_valid))
          flog.write("Valid: {}, step: {}, loss: {}, mean loss: {}\n".format(epoch, step//args.batch_size, loss_batch, mean_loss_valid))

        # plot_result(epoch, args.num_plots, figure_dir,  preds_batch, X_batch, Y_batch)
        pool.map(partial(plot_result_thread, 
                         epoch = epoch,
                         preds = preds_batch,
                         X = X_batch,
                         Y = Y_batch,
                         figure_dir = figure_dir),
                range(args.num_plots))

    flog.close()
    pool.close()
    data_reader.coord.request_stop()
    if data_reader_valid is not None:
      data_reader_valid.coord.request_stop()
    try:
      data_reader.coord.join(threads, stop_grace_period_secs=10, ignore_live_threads=True)
      if data_reader_valid is not None:
        data_reader_valid.coord.join(threads_valid, stop_grace_period_secs=10, ignore_live_threads=True)
    except:
      pass
    sess.run(data_reader.queue.close(cancel_pending_enqueues=True))
    if data_reader_valid is not None:
      sess.run(data_reader_valid.queue.close(cancel_pending_enqueues=True))
  return 0


def test_fn(args, data_reader, figure_dir=None, result_dir=None):
  current_time = time.strftime("%y%m%d-%H%M%S")
  log_dir = os.path.join(args.log_dir, args.mode, current_time)
  logging.info("{} log: {}".format(args.mode, log_dir))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if (args.plot_figure == True) and (figure_dir is None):
    figure_dir = os.path.join(log_dir, 'figures')
    if not os.path.exists(figure_dir):
      os.makedirs(figure_dir)
  if (args.save_result == True) and (result_dir is None):
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  config = set_config(args, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.compat.v1.name_scope('Input_Batch'):
    batch = data_reader.dequeue(args.batch_size)

  model = UNet(config, input_batch=batch, mode='test')
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.compat.v1.Session(config=sess_config) as sess:

    summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(args.model_dir)
    saver.restore(sess, latest_check_point)

    threads = data_reader.start_threads(sess, n_threads=multiprocessing.cpu_count())

    flog = open(os.path.join(log_dir, 'loss.log'), 'w')
    total_step = 0
    mean_loss = 0
    progressbar = tqdm(range(0, data_reader.n_signal, args.batch_size), desc=args.mode)
    if args.plot_figure:                                                           
      num_pool = multiprocessing.cpu_count()*2                                     
    elif args.save_result:                                                         
      num_pool = multiprocessing.cpu_count()                                       
    else:                                                                          
      num_pool = 2                                                                 
    pool = multiprocessing.Pool(num_pool) 
    for step in progressbar:

      if step + args.batch_size >= data_reader.n_signal:
        for t in threads:
          t.join()
        sess.run(data_reader.queue.close())
      
      loss_batch, preds_batch, X_batch, Y_batch, ratio_batch, \
      signal_batch, noise_batch, fname_batch = model.test_on_batch(sess, summary_writer)
      total_step += 1
      mean_loss += (loss_batch-mean_loss)/total_step
      progressbar.set_description("{}: loss={:.6f}, mean loss={:6f}".format(args.mode, loss_batch, mean_loss))
      flog.write("step: {}, loss: {}\n".format(step, loss_batch))
      flog.flush()

      pool.map(partial(postprocessing_test, 
                      preds=preds_batch, 
                      X=X_batch*ratio_batch[:,np.newaxis,np.newaxis,np.newaxis],
                      fname=fname_batch,
                      figure_dir=figure_dir,
                      result_dir=result_dir,
                      signal_FT=signal_batch, 
                      noise_FT=noise_batch), 
                range(len(X_batch)))

    flog.close()
    pool.close()

  return 0

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
  if args.save_result:
    result_dir = os.path.join(log_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

  config = set_config(args, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.compat.v1.name_scope('Input_Batch'):
   data_batch = data_reader.dataset(args.batch_size)

  # model = UNet(config, input_batch=batch, mode='pred')
  model = UNet(config, mode='pred')
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  #sess_config.log_device_placement = False

  with tf.compat.v1.Session(config=sess_config) as sess:

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(args.model_dir)
    saver.restore(sess, latest_check_point)

#    threads = data_reader.start_threads(sess, n_threads=multiprocessing.cpu_count())

    if args.plot_figure:                                                           
      num_pool = multiprocessing.cpu_count()                                   
    elif args.save_result:                                                         
      num_pool = multiprocessing.cpu_count()                                       
    else:                                                                          
      num_pool = 2                                                                 
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(num_pool)
    for step in tqdm(range(0, data_reader.n_signal, args.batch_size), desc="Pred"):
      #if step + args.batch_size >= data_reader.n_signal:
      #  for t in threads:
      #    t.join()
      #  sess.run(data_reader.queue.close())
      # X_batch = []
      # ratio_batch = []
      # fname_batch = []
      # for i in range(step, min(step+args.batch_size, data_reader.n_signal)):
      #   X, ratio, fname = data_reader[i]
      #   if np.std(X) == 0:
      #     continue
      #   X_batch.append(X)
      #   ratio_batch.append(ratio)
      #   fname_batch.append(fname)
      # X_batch = np.stack(X_batch, axis=0)
      # ratio_batch = np.array(ratio_batch)
      X_batch, ratio_batch, fname_batch = sess.run(data_batch)
      preds_batch = sess.run(model.preds, feed_dict={model.X: X_batch,
                                                     model.drop_rate: 0,
                                                     model.is_training: False})
      #preds_batch, X_batch, ratio_batch, fname_batch = sess.run([model.preds,
      #                                                    batch[0],
      #                                                    batch[1],
      #                                                    batch[2]],
      #                                                    feed_dict={model.drop_rate: 0,
      #                                                               model.is_training: False})

      pool.map(partial(postprocessing_pred,
                       preds = preds_batch, 
                       X = X_batch*ratio_batch[:,np.newaxis,:,np.newaxis],
                       fname = [x.decode() for x in fname_batch],
                       figure_dir = figure_dir, 
                       result_dir = result_dir), 
               range(len(X_batch)))

      # for i in range(len(X_batch)):
      #   postprocessing_thread(i,
      #             preds = preds_batch, 
      #             X = X_batch*ratio_batch[:,np.newaxis,np.newaxis,np.newaxis],
      #             fname = fname_batch,
      #             figure_dir = figure_dir, 
      #             result_dir = result_dir)
    
    pool.close()

  return 0

def main(args):

  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

  coord = tf.train.Coordinator()

  if args.mode == "train":
    with tf.compat.v1.name_scope('create_inputs'):
      data_reader = DataReader(
          signal_dir=args.train_signal_dir,
          signal_list=args.train_signal_list,
          noise_dir=args.train_noise_dir,
          noise_list=args.train_noise_list,
          queue_size=args.batch_size*2,
          coord=coord)
      if (args.valid_signal_list is not None) and (args.valid_noise_list is not None):
        data_reader_valid = DataReader(
            signal_dir=args.valid_signal_dir,
            signal_list=args.valid_signal_list,
            noise_dir=args.valid_noise_dir,
            noise_list=args.valid_noise_list,
            queue_size=args.batch_size*2,
            coord=coord)
        logging.info("Dataset size: training %d, validation %d" %  (data_reader.n_signal, data_reader_valid.n_signal))
      else:
        data_reader_valid = None
      logging.info("Dataset size: training %d, validation 0" %  (data_reader.n_signal))
    train_fn(args, data_reader, data_reader_valid)
  
  elif args.mode == "valid" or args.mode == "test":
    with tf.compat.v1.name_scope('create_inputs'):
      data_reader = DataReader_test(
          signal_dir=args.valid_signal_dir,
          signal_list=args.valid_signal_list,
          noise_dir=args.valid_noise_dir,
          noise_list=args.valid_noise_list,
          queue_size=args.batch_size*2,
          coord=coord)
    logging.info("Dataset Size: {}".format(data_reader.n_signal))
    test_fn(args, data_reader)

  elif args.mode == "pred":
    with tf.compat.v1.name_scope('create_inputs'):
      data_reader = DataReader_pred(
          signal_dir=args.data_dir,
          signal_list=args.data_list,
          sampling_rate=args.sampling_rate)
    logging.info("Dataset Size: {}".format(data_reader.n_signal))
    pred_fn(args, data_reader, log_dir=args.output_dir)

  else:
    print("mode should be: train, valid, test, debug or pred")

  coord.request_stop()
  coord.join()
  return 0

if __name__ == '__main__':
  args = read_args()
  main(args)
