# --------------------------------------------------------
# Sequence Level Semantics Aggregation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# Modified by Haiping Wu
# --------------------------------------------------------

import mxnet as mx
import numpy as np

class _BinaryClassificationMetrics(object):
    """
    Private container class for classification metric statistics. True/false positive and
     true/false negative counts are sufficient statistics for various classification metrics.
    This class provides the machinery to track those statistics across mini-batches of
    (label, prediction) pairs.
    """

    def __init__(self):
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.true_negatives = 0

    def update_binary_stats(self, label, pred_label):
        """
        Update various binary classification counts for a single (label, pred)
        pair.

        Parameters
        ----------
        """
        # check_label_shapes(label, pred)
        if len(np.unique(label)) > 2:
            raise ValueError("%s currently only supports binary classification."
                             % self.__class__.__name__)
        pred_true = (pred_label == 1)
        pred_false = 1 - pred_true
        label_true = (label == 1)
        label_false = 1 - label_true

        self.true_positives += (pred_true * label_true).sum()
        self.false_positives += (pred_true * label_false).sum()
        self.false_negatives += (pred_false * label_true).sum()
        self.true_negatives += (pred_false * label_false).sum()

    @property
    def precision(self):
        if self.true_positives + self.false_positives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_positives)
        else:
            return 0.

    @property
    def recall(self):
        if self.true_positives + self.false_negatives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_negatives)
        else:
            return 0.

    @property
    def fscore(self):
        if self.precision + self.recall > 0:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            return 0.

    @property
    def matthewscc(self):
        """
        Calculate the Matthew's Correlation Coefficent
        """
        if not self.total_examples:
            return 0.

        true_pos = float(self.true_positives)
        false_pos = float(self.false_positives)
        false_neg = float(self.false_negatives)
        true_neg = float(self.true_negatives)
        terms = [(true_pos + false_pos),
                 (true_pos + false_neg),
                 (true_neg + false_pos),
                 (true_neg + false_neg)]
        denom = 1.
        for t in filter(lambda t: t != 0., terms):
            denom *= t
        return ((true_pos * true_neg) - (false_pos * false_neg)) / math.sqrt(denom)

    @property
    def total_examples(self):
        return self.false_negatives + self.false_positives + \
               self.true_negatives + self.true_positives

    def reset_stats(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0

def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if cfg.TRAIN.ENABLE_OHEM or cfg.TRAIN.END2END:
        pred.append('rcnn_label')
    if cfg.TRAIN.END2END:
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


def get_sim_names(cfg):
    pred = ['sim_label_prob']
    label = ['sim_label']

    rcnn_pred, rcnn_label = get_rcnn_names(cfg)
    pred = rcnn_pred + pred
    label = rcnn_label + label

    return pred, label

class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        # print(pred.dtype, label.dtype)
        # print(pred)
        # b = pred.asnumpy()
        # pred_label = np.argmax(pred.asnumpy(), axis=1).astype('int32')
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class SimAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(SimAccMetric, self).__init__('SimAcc')
        self.pred, self.label = get_sim_names(cfg)

    def update(self, labels, preds):
        pred = preds[6].asnumpy()
        label = preds[5].asnumpy().astype('int32')


        label_index = np.where(label != -1)[1]
        pred = pred[:, label_index]
        label = label[:, label_index]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = (pred > 0.5).astype('int32')
        pred_label = pred_label.reshape((1, -1))


        label = label.reshape((1, -1))
        # label (b, p)
        label = label.astype('int32')

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class SimTpMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(SimTpMetric, self).__init__('SimTp')
        self.pred, self.label = get_sim_names(cfg)
        self.metrics = _BinaryClassificationMetrics()

    def update(self, labels, preds):
        pred = preds[6].asnumpy()
        label = preds[5].asnumpy().astype('int32')

        label_index = np.where(label >= 0)[1]
        pred = pred[:, label_index]
        label = label[:, label_index]

        # positive_label_index = np.where(label == 1)[1]
        # negative_label_index = np.where(label == 0)[1]
        # pred (b, c, p) or (b, c, h, w)
        pred_label = (pred > 0.5).astype('int32')
        pred_label = pred_label.reshape((1, -1))

        label = label.reshape((1, -1))

        self.metrics.update_binary_stats(label, pred_label)
        self.sum_metric += self.metrics.true_positives
        self.num_inst += self.metrics.true_positives + self.metrics.false_negatives
        self.metrics.reset_stats()

    # def reset(self):
    #     self.sum_metric = 0.
    #     self.num_inst = 0.
    #     self.metrics.reset_stats()

class SimTnMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(SimTnMetric, self).__init__('SimTn')
        self.pred, self.label = get_sim_names(cfg)
        self.metrics = _BinaryClassificationMetrics()

    def update(self, labels, preds):
        pred = preds[6].asnumpy()
        label = preds[5].asnumpy().astype('int32')


        label_index = np.where(label >= 0)[1]
        pred = pred[:, label_index]
        label = label[:, label_index]

        # label_index = np.where(label <= -2)[1]
        # pred = pred[:, label_index]
        # label = label[:, label_index] + 2


        # pred (b, c, p) or (b, c, h, w)
        pred_label = (pred > 0.5).astype('int32')
        pred_label = pred_label.reshape((1, -1))

        self.metrics.update_binary_stats(label, pred_label)
        self.sum_metric += self.metrics.true_negatives
        self.num_inst += self.metrics.true_negatives + self.metrics.false_positives
        self.metrics.reset_stats()

    # def reset(self):
    #     self.sum_metric = 0.
    #     self.num_inst = 0.
    #     self.metrics.reset_stats()

class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class TripletHardLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(TripletHardLossMetric, self).__init__('TripletHardLoss')

    def update(self, labels, preds):
        triplet_hard_loss = preds[5].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = 1

        self.sum_metric += np.sum(triplet_hard_loss)
        self.num_inst += num_inst
