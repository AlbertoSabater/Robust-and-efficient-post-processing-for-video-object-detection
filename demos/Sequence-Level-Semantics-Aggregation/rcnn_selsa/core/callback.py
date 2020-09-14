# --------------------------------------------------------
# Sequence Level Semantics Aggregation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong, Haiping Wu
# --------------------------------------------------------

import time
import logging
import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)

                logging.info(s)
                print(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        if 'rfcn_bbox_weight' in arg:
            weight = arg['rfcn_bbox_weight']
            bias = arg['rfcn_bbox_bias']
            repeat = bias.shape[0] / means.shape[0]

            arg['rfcn_bbox_weight_test'] = weight * mx.nd.repeat(mx.nd.array(stds), repeats=repeat).reshape((
                bias.shape[0], 1, 1, 1))
            arg['rfcn_bbox_bias_test'] = arg['rfcn_bbox_bias'] * mx.nd.repeat(mx.nd.array(stds), repeats=repeat) \
                                         + mx.nd.repeat(mx.nd.array(means), repeats=repeat)
            mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
            arg.pop('rfcn_bbox_weight_test')
            arg.pop('rfcn_bbox_bias_test')
        elif 'bbox_pred_weight' in arg:
            arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
            arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
            mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
            arg.pop('bbox_pred_weight_test')
            arg.pop('bbox_pred_bias_test')

    return _callback


class TensorboardCallback(object):
    """Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`,
    but write TensorBoard event file for visualization.
    For more usage, please refer https://github.com/dmlc/tensorboard
    Parameters
    ----------
    logging_dir : str
        TensorBoard event file directory.
        After that, use `tensorboard --logdir=path/to/logs` to launch
        TensorBoard visualization.
    prefix : str
        Prefix for a metric name of `scalar` value.
        You might want to use this param to leverage TensorBoard plot feature,
        where TensorBoard plots different curves in one graph when they have
          same `name`.
        The follow example shows the usage(how to compare a train and eval
          metric in a same graph).
    """

    def __init__(self, logging_dir, total_step=0, prefix=None):
        self.prefix = prefix
        self.step = total_step
        try:
            from mxboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error(
                'You can install tensorboard via `pip install mxboard`.')

    def __call__(self, param, name_value=None):
        """Callback to log training speed and metrics in TensorBoard."""
        if name_value:
            self._add_scalar(name_value)

        if param.eval_metric is None:
            return

        # if param.add_step:
        self.step += 1

        name_value = param.eval_metric.get_name_value()
        if name_value:
            self._add_scalar(name_value)

    def _add_scalar(self, name_value):
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(tag=name, value=value, global_step=self.step)
