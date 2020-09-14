# --------------------------------------------------------
# Sequence Level Semantics Aggregation
# Copyright (c) 2017 Microsoft
# Copyright (c) 2019 by Contributors
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang, Bin Xiao
# Modified by Haiping Wu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.rpn_inv_normalize import *
from operator_py.batch_norm import BatchNorm
import math

class resnet_v1_101_rcnn_selsa(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]


    def get_resnet_v1_conv4(self, data, config, dtype='float32'):
        # dtype = 'float16'
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        # bn_conv1 = BatchNorm(config, name='bn_conv1', data=conv1, use_global_stats=self.use_global_stats,
        #                                eps=self.eps, fix_gamma=False)
        bn_conv1 = BatchNorm(config, name='bn_conv1', data=conv1, use_global_stats=self.use_global_stats,
                                       eps=self.eps, fix_gamma=False)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                  pool_type='max')


        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch1 = BatchNorm(config, name='bn2a_branch1', data=res2a_branch1,
                                           use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch1 = bn2a_branch1 # residual branch


        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2a = BatchNorm(config, name='bn2a_branch2a', data=res2a_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = BatchNorm(config, name='bn2a_branch2b', data=res2a_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = BatchNorm(config, name='bn2a_branch2c', data=res2a_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = BatchNorm(config, name='bn2b_branch2a', data=res2b_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = BatchNorm(config, name='bn2b_branch2b', data=res2b_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = BatchNorm(config, name='bn2b_branch2c', data=res2b_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = BatchNorm(config, name='bn2c_branch2a', data=res2c_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = BatchNorm(config, name='bn2c_branch2b', data=res2c_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = BatchNorm(config, name='bn2c_branch2c', data=res2c_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = BatchNorm(config, name='bn3a_branch1', data=res3a_branch1,
                                           use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = BatchNorm(config, name='bn3a_branch2a', data=res3a_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = BatchNorm(config, name='bn3a_branch2b', data=res3a_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = BatchNorm(config, name='bn3a_branch2c', data=res3a_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = BatchNorm(config, name='bn3b1_branch2a', data=res3b1_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = BatchNorm(config, name='bn3b1_branch2b', data=res3b1_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = BatchNorm(config, name='bn3b1_branch2c', data=res3b1_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = BatchNorm(config, name='bn3b2_branch2a', data=res3b2_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = BatchNorm(config, name='bn3b2_branch2b', data=res3b2_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = BatchNorm(config, name='bn3b2_branch2c', data=res3b2_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = BatchNorm(config, name='bn3b3_branch2a', data=res3b3_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = BatchNorm(config, name='bn3b3_branch2b', data=res3b3_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = BatchNorm(config, name='bn3b3_branch2c', data=res3b3_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = BatchNorm(config, name='bn4a_branch1', data=res4a_branch1,
                                           use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = BatchNorm(config, name='bn4a_branch2a', data=res4a_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = BatchNorm(config, name='bn4a_branch2b', data=res4a_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = BatchNorm(config, name='bn4a_branch2c', data=res4a_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = BatchNorm(config, name='bn4b1_branch2a', data=res4b1_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = BatchNorm(config, name='bn4b1_branch2b', data=res4b1_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = BatchNorm(config, name='bn4b1_branch2c', data=res4b1_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = BatchNorm(config, name='bn4b2_branch2a', data=res4b2_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = BatchNorm(config, name='bn4b2_branch2b', data=res4b2_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = BatchNorm(config, name='bn4b2_branch2c', data=res4b2_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = BatchNorm(config, name='bn4b3_branch2a', data=res4b3_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = BatchNorm(config, name='bn4b3_branch2b', data=res4b3_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = BatchNorm(config, name='bn4b3_branch2c', data=res4b3_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = BatchNorm(config, name='bn4b4_branch2a', data=res4b4_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = BatchNorm(config, name='bn4b4_branch2b', data=res4b4_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = BatchNorm(config, name='bn4b4_branch2c', data=res4b4_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = BatchNorm(config, name='bn4b5_branch2a', data=res4b5_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = BatchNorm(config, name='bn4b5_branch2b', data=res4b5_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = BatchNorm(config, name='bn4b5_branch2c', data=res4b5_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = BatchNorm(config, name='bn4b6_branch2a', data=res4b6_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = BatchNorm(config, name='bn4b6_branch2b', data=res4b6_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = BatchNorm(config, name='bn4b6_branch2c', data=res4b6_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = BatchNorm(config, name='bn4b7_branch2a', data=res4b7_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = BatchNorm(config, name='bn4b7_branch2b', data=res4b7_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = BatchNorm(config, name='bn4b7_branch2c', data=res4b7_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = BatchNorm(config, name='bn4b8_branch2a', data=res4b8_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = BatchNorm(config, name='bn4b8_branch2b', data=res4b8_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = BatchNorm(config, name='bn4b8_branch2c', data=res4b8_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = BatchNorm(config, name='bn4b9_branch2a', data=res4b9_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = BatchNorm(config, name='bn4b9_branch2b', data=res4b9_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = BatchNorm(config, name='bn4b9_branch2c', data=res4b9_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = BatchNorm(config, name='bn4b10_branch2a', data=res4b10_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = BatchNorm(config, name='bn4b10_branch2b', data=res4b10_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = BatchNorm(config, name='bn4b10_branch2c', data=res4b10_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = BatchNorm(config, name='bn4b11_branch2a', data=res4b11_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = BatchNorm(config, name='bn4b11_branch2b', data=res4b11_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = BatchNorm(config, name='bn4b11_branch2c', data=res4b11_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = BatchNorm(config, name='bn4b12_branch2a', data=res4b12_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = BatchNorm(config, name='bn4b12_branch2b', data=res4b12_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = BatchNorm(config, name='bn4b12_branch2c', data=res4b12_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = BatchNorm(config, name='bn4b13_branch2a', data=res4b13_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = BatchNorm(config, name='bn4b13_branch2b', data=res4b13_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = BatchNorm(config, name='bn4b13_branch2c', data=res4b13_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = BatchNorm(config, name='bn4b14_branch2a', data=res4b14_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = BatchNorm(config, name='bn4b14_branch2b', data=res4b14_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = BatchNorm(config, name='bn4b14_branch2c', data=res4b14_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = BatchNorm(config, name='bn4b15_branch2a', data=res4b15_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = BatchNorm(config, name='bn4b15_branch2b', data=res4b15_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = BatchNorm(config, name='bn4b15_branch2c', data=res4b15_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = BatchNorm(config, name='bn4b16_branch2a', data=res4b16_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = BatchNorm(config, name='bn4b16_branch2b', data=res4b16_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = BatchNorm(config, name='bn4b16_branch2c', data=res4b16_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = BatchNorm(config, name='bn4b17_branch2a', data=res4b17_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = BatchNorm(config, name='bn4b17_branch2b', data=res4b17_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = BatchNorm(config, name='bn4b17_branch2c', data=res4b17_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = BatchNorm(config, name='bn4b18_branch2a', data=res4b18_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = BatchNorm(config, name='bn4b18_branch2b', data=res4b18_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = BatchNorm(config, name='bn4b18_branch2c', data=res4b18_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = BatchNorm(config, name='bn4b19_branch2a', data=res4b19_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = BatchNorm(config, name='bn4b19_branch2b', data=res4b19_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = BatchNorm(config, name='bn4b19_branch2c', data=res4b19_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = BatchNorm(config, name='bn4b20_branch2a', data=res4b20_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = BatchNorm(config, name='bn4b20_branch2b', data=res4b20_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = BatchNorm(config, name='bn4b20_branch2c', data=res4b20_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = BatchNorm(config, name='bn4b21_branch2a', data=res4b21_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = BatchNorm(config, name='bn4b21_branch2b', data=res4b21_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = BatchNorm(config, name='bn4b21_branch2c', data=res4b21_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = BatchNorm(config, name='bn4b22_branch2a', data=res4b22_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b22_branch2b = BatchNorm(config, name='bn4b22_branch2b', data=res4b22_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = BatchNorm(config, name='bn4b22_branch2c', data=res4b22_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')

        if dtype == 'float16':
            res4b22_relu = mx.sym.Cast(data=res4b22_relu, dtype=np.float32)

        return res4b22_relu
    def get_resnet_v1_conv5(self, conv_feat, config, dtype='float32'):
        # dtype = 'float16'
        if dtype == 'float16':
            conv_feat = mx.sym.Cast(data=conv_feat, dtype=np.float16)
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = BatchNorm(config, name='bn5a_branch1', data=res5a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1 # residual branch



        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = BatchNorm(config, name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512,
                                               pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5a_branch2b = BatchNorm(config, name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = BatchNorm(config, name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')


        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = BatchNorm(config, name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512,
                                               pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5b_branch2b = BatchNorm(config, name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = BatchNorm(config, name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')


        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = BatchNorm(config, name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512,
                                               pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5c_branch2b = BatchNorm(config, name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = BatchNorm(config, name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')

        ##################
        # cast type
        if dtype == 'float16':
            res5c_relu = mx.sym.Cast(data=res5c_relu, dtype=np.float32)
        ##################

        return res5c_relu

    def get_rpn(self, conv_feat, num_anchors, dtype='float32'):
        ###########
        if dtype == 'float16':
            conv_feat = mx.sym.Cast(data=conv_feat, dtype=np.float32)
        ###########
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def semantic_aggregation(self,  roi_feat,
                             nongt_dim, feat_dim,
                             non_cur_space=False,
                             dim=(1024, 1024, 1024),
                             index=1,
                             key_dim=0,
                             t_dim=1,
                             output_cur_only=False,
                             conv_z=True,
                             conv_g=False,
                             ):
        print('enter aggregation module')
        nongt_roi_feat = mx.symbol.slice_axis(data=roi_feat, axis=0, begin=0, end=nongt_dim)

        if non_cur_space:
            nongt_roi_feat_split = mx.sym.SliceChannel(data=nongt_roi_feat, axis=0, num_outputs=t_dim)
            if key_dim != 0:
                cur_nongt_roi_feat = mx.sym.Concat(mx.sym.Concat(*nongt_roi_feat_split[0:key_dim], dim=0),
                                                   mx.sym.Concat(*nongt_roi_feat_split[key_dim+1:], dim=0),
                                                   dim=0)
            else:
                cur_nongt_roi_feat = mx.sym.Concat(*nongt_roi_feat_split[key_dim+1:], dim=0)
            nongt_roi_feat = cur_nongt_roi_feat

        #########
        if output_cur_only:
            roi_feat = mx.sym.SliceChannel(roi_feat, axis=0, num_outputs=t_dim)[key_dim]
        ########

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        q_data = mx.sym.FullyConnected(name='query_' + str(index),
                                       data=roi_feat,
                                       num_hidden=dim[0])
        q_data_batch = mx.sym.Reshape(q_data, shape=(-1, 1, dim[0]))
        q_data_batch = mx.sym.transpose(q_data_batch, axes=(1, 0, 2))
        k_data = mx.symbol.FullyConnected(name='key_' + str(index),
                                          data=nongt_roi_feat,
                                          num_hidden=dim[1])
        k_data_batch = mx.sym.Reshape(k_data, shape=(-1, 1, dim[1]))
        k_data_batch = mx.sym.transpose(k_data_batch, axes=(1, 0, 2))
        v_data = nongt_roi_feat
        aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(dim[1]))) * aff
        aff_scale = mx.sym.transpose(aff_scale, axes=(1, 0, 2))

        weighted_aff = aff_scale

        aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=2, name='softmax_' + str(index))

        similarity = mx.sym.identity(aff_softmax)

        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(-3, -2))
        # output_t, [num_rois * fc_dim, feat_dim]
        if conv_g:
            v_data = mx.sym.FullyConnected(name='value_'+str(index),
                                           data=v_data,
                                           num_hidden=dim[2])
        output_t = mx.symbol.dot(lhs=aff_softmax_reshape, rhs=v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = mx.sym.Reshape(output_t, shape=(-1, feat_dim, 1, 1))
        # linear_out, [num_rois, dim[2], 1, 1]

        if conv_z:
            linear_out = mx.symbol.Convolution(name='linear_out_' + str(index), data=output_t,
                                               kernel=(1, 1), num_filter=dim[2], num_group=1)
        else:
            linear_out = output_t

        output = mx.sym.Reshape(linear_out, shape=(0, 0))

        return output, similarity

    def get_train_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS
        dtype = cfg.network.dtype

        data = mx.sym.Variable(name="data")
        data_bef = mx.sym.Variable(name="data_bef")
        data_aft = mx.sym.Variable(name="data_aft")
        im_info = mx.sym.Variable(name="im_info")
        gt_boxes = mx.sym.Variable(name="gt_boxes")
        rpn_label = mx.sym.Variable(name='label')
        rpn_bbox_target = mx.sym.Variable(name='bbox_target')
        rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

        grad_scale_factor = 1.0

        data_range = cfg.TRAIN.train_frames
        concat_data = mx.sym.concat(*[data, data_bef, data_aft], dim=0)

        # shared convolutional layers
        conv_feat = self.get_resnet_v1_conv4(concat_data, config=cfg, dtype=dtype)
        # res5
        relu1 = self.get_resnet_v1_conv5(conv_feat, config=cfg, dtype=dtype)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)

        # prepare rpn data
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        #######################################
        # split
        rpn_cls_score_reshape_split = mx.sym.split(rpn_cls_score_reshape, axis=0, num_outputs=data_range)
        rpn_bbox_pred_split = mx.sym.split(rpn_bbox_pred, axis=0, num_outputs=data_range)
        #######################################

        # classification

        rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape_split[0], label=rpn_label, multi_output=True,
                                            normalization='valid', use_ignore=True, ignore_label=-1,
                                            name="rpn_cls_prob", grad_scale=grad_scale_factor)

        # bounding box regression
        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0,
                                                                data=(rpn_bbox_pred_split[0]- rpn_bbox_target))
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)
            rpn_bbox_pred_split = mx.sym.split(rpn_bbox_pred, axis=0, num_outputs=data_range)
        else:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                                data=(rpn_bbox_pred_split[0]- rpn_bbox_target))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                        grad_scale=1.0 * grad_scale_factor / cfg.TRAIN.RPN_BATCH_SIZE)

        # ROI proposal
        rpn_cls_act = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
        rpn_cls_act_reshape = mx.sym.Reshape(
            data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

        ###############################
        rois = []
        rpn_cls_act_reshape_split = mx.sym.split(rpn_cls_act_reshape, axis=0, num_outputs=data_range)
        ###############################

        for i in range(data_range):
            if cfg.TRAIN.CXX_PROPOSAL:
                _rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape_split[i], bbox_pred=rpn_bbox_pred_split[i], im_info=im_info,
                    name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                _rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape_split[i], bbox_pred=rpn_bbox_pred_split[i], im_info=im_info,
                    name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            rois.append(_rois)
        # ROI proposal target
        gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')


        rois_all = []
        rois_cur, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois[0], gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  append_gt=cfg.TRAIN.append_gt,
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        rois_all.append(rois_cur)
        for i in range(1, data_range):
            _rois, _label, _bbox_target, _bbox_weight = mx.sym.Custom(rois=rois[i], gt_boxes=gt_boxes_reshape,
                                                                      op_type='proposal_target',
                                                                      num_classes=num_reg_classes,
                                                                      batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                      batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                      cfg=cPickle.dumps(cfg),
                                                                      append_gt=cfg.TRAIN.append_gt,
                                                                      fg_fraction=cfg.TRAIN.FG_FRACTION)
            rois_all.append(_rois)


        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        roi_pool_all = []
        conv_new_1_relu_split = mx.sym.split(conv_new_1_relu, axis=0, num_outputs=data_range)
        for i in range(data_range):
            _roi_pool = mx.symbol.ROIPooling(
                name='roi_pool', data=conv_new_1_relu_split[i], rois=rois_all[i], pooled_size=(7, 7), spatial_scale=0.0625)
            roi_pool_all.append(_roi_pool)
        roi_pool = mx.sym.concat(*roi_pool_all, dim=0)

        ##################################################
        nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N * data_range
        rois = mx.sym.concat(*rois_all, dim=0)

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
        # attention, [num_rois, feat_dim]
        ###
        attention_1, _ = self.semantic_aggregation(fc_new_1,
                                                   nongt_dim=nongt_dim, feat_dim=1024,
                                                   index=1,
                                                   dim=(cfg.RELATION.inter_filter, cfg.RELATION.inter_filter, 1024),
                                                   conv_g=cfg.RELATION.conv_g,
                                                   conv_z=cfg.RELATION.conv_z,
                                                   non_cur_space=cfg.RELATION.non_cur_space,
                                                   t_dim=data_range,
                                                   key_dim=0,
                                                   )
        fc_all_1 = fc_new_1 + attention_1
        fc_all_1_relu = mx.sym.Activation(data=fc_all_1, act_type='relu', name='fc_all_1_relu')

        output_cur_only = cfg.RELATION.relation_output_cur_only
        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_all_1_relu, num_hidden=1024)
        attention_2, _ = self.semantic_aggregation(fc_new_2,
                                                   nongt_dim=nongt_dim, feat_dim=1024,
                                                   index=2,
                                                   conv_g=cfg.RELATION.conv_g,
                                                   conv_z=cfg.RELATION.conv_z,
                                                   non_cur_space=cfg.RELATION.non_cur_space,
                                                   t_dim=data_range,
                                                   key_dim=0,
                                                   output_cur_only=output_cur_only,
                                                   dim=(cfg.RELATION.inter_filter, cfg.RELATION.inter_filter, 1024))

        if output_cur_only:
            fc_new_2 = mx.sym.split(fc_new_2, axis=0, num_outputs=data_range)[0]

        fc_all_2 = fc_new_2 + attention_2

        ############
        if not output_cur_only:
            fc_all_2 = mx.sym.split(fc_all_2, axis=0, num_outputs=data_range)[0]
        ############
        fc_all_2_relu = mx.sym.Activation(data=fc_all_2, act_type='relu', name='fc_all_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_all_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_all_2_relu, num_hidden=num_reg_classes * 4)
        ####################################################

        if cfg.TRAIN.ENABLE_OHEM:
            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes,
                                                           roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                            normalization='valid', use_ignore=True, ignore_label=-1,
                                            grad_scale=grad_scale_factor)
            bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                              data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                        grad_scale=1.0 * grad_scale_factor / cfg.TRAIN.BATCH_ROIS_OHEM)
            rcnn_label = labels_ohem
        else:
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                            grad_scale=grad_scale_factor)
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                        data=(bbox_pred - bbox_target))

            if cfg.TRAIN.BATCH_ROIS < 0:
                batch_rois_num = 300
            else:
                batch_rois_num = cfg.TRAIN.BATCH_ROIS

            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 * grad_scale_factor
                                                                                      / batch_rois_num)
            rcnn_label = label

        # reshape output
        rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                  name='cls_prob_reshape')
        bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                   name='bbox_loss_reshape')


        group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)
                              ])
        self.sym = group
        return group

    def get_feat_symbol(self, cfg):
        # config alias for convenient
        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")
        data_cache = mx.sym.Variable(name="data_cache")
        feat_cache = mx.sym.Variable(name="feat_cache")

        dtype = cfg.network.dtype
        conv_feat = self.get_resnet_v1_conv4(data, config=cfg, dtype=dtype)
        conv_embed = mx.sym.identity(conv_feat, name="conv_embed")
        group = mx.sym.Group([conv_embed, im_info, data_cache, feat_cache])
        self.sym = group
        return group

    def get_aggregation_symbol(self, cfg):
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS
        data_range = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1

        dtype = cfg.network.dtype


        data_cur = mx.sym.Variable(name="data")                 # not used
        im_info = mx.sym.Variable(name="im_info")
        data_cache = mx.sym.Variable(name="data_cache")         # data_cache contains data_range images
        feat_cache = mx.sym.Variable(name="feat_cache")         # feat_cache contains the data_range
                                                                # feature maps of the images

        conv_feat = mx.sym.slice_axis(feat_cache, axis=1, begin=0, end=1024)

        # conv5
        relu1 = self.get_resnet_v1_conv5(conv_feat, config=cfg, dtype=dtype)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors, dtype=dtype)

        # prepare rpn data
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI proposal
        rpn_cls_act = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
        rpn_cls_act_reshape = mx.sym.Reshape(
            data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

        if cfg.TRAIN.CXX_PROPOSAL:
            rois = mx.contrib.sym.MultiProposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred,
                im_info=mx.sym.tile(im_info, reps=(data_range, 1)),
                name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)


        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool', data=conv_new_1_relu, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)

        nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N * data_range

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
        # attention, [num_rois, feat_dim]

        # if data_range equals one, no transformation is used.
        if data_range != 1 or (cfg.TEST.single_frame_trans):
            attention_1, similarity_1 = \
                self.semantic_aggregation(fc_new_1,
                                          nongt_dim,
                                          feat_dim=1024,
                                          index=1,
                                          conv_g=cfg.RELATION.conv_g,
                                          conv_z=cfg.RELATION.conv_z,
                                          non_cur_space=cfg.RELATION.non_cur_space,
                                          t_dim=data_range,
                                          key_dim=cfg.TEST.KEY_FRAME_INTERVAL,
                                          dim=(cfg.RELATION.inter_filter, cfg.RELATION.inter_filter, 1024))
            fc_all_1 = fc_new_1 + attention_1
        else:
            # no transformation
            print('no transformation')
            fc_all_1 = fc_new_1

        fc_all_1_relu = mx.sym.Activation(data=fc_all_1, act_type='relu', name='fc_all_1_relu')

        output_cur_only= cfg.RELATION.relation_output_cur_only
        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_all_1_relu, num_hidden=1024)


        if data_range != 1 or (cfg.TEST.single_frame_trans):
            attention_2, similarity_2 = \
                self.semantic_aggregation(fc_new_2,
                                          nongt_dim=nongt_dim, feat_dim=1024,
                                          index=2,
                                          conv_g=cfg.RELATION.conv_g,
                                          conv_z=cfg.RELATION.conv_z,
                                          non_cur_space=cfg.RELATION.non_cur_space,
                                          t_dim=data_range,
                                          key_dim=cfg.TEST.KEY_FRAME_INTERVAL,
                                          output_cur_only=output_cur_only,
                                          dim=(cfg.RELATION.inter_filter, cfg.RELATION.inter_filter, 1024)
                                          )

            if output_cur_only:
                fc_new_2 = mx.sym.split(fc_new_2, axis=0, num_outputs=data_range)[cfg.TEST.KEY_FRAME_INTERVAL]

            fc_all_2 = fc_new_2 + attention_2

            if not output_cur_only:
                fc_all_2 = mx.sym.split(fc_all_2, axis=0, num_outputs=data_range)[cfg.TEST.KEY_FRAME_INTERVAL]
        else:
            # no transformation
            fc_all_2 = fc_new_2

        fc_all_2_relu = mx.sym.Activation(data=fc_all_2, act_type='relu', name='fc_all_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_all_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_all_2_relu, num_hidden=num_reg_classes * 4)

        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                  name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                   name='bbox_pred_reshape')
        rois = mx.sym.split(rois, axis=0, num_outputs=data_range)[cfg.TEST.KEY_FRAME_INTERVAL]
        rois = mx.sym.identity(rois, name='rois')

        if data_range != 1:
            similarity = mx.sym.identity(attention_1, name='nonlocal_weights')
            group = mx.sym.Group([data_cur, rois, cls_prob, bbox_pred, data_cache, similarity])
        else:
            group = mx.sym.Group([data_cur, rois, cls_prob, bbox_pred, data_cache])

        self.sym = group
        return group

    def init_weight_selsa(self, cfg, arg_params, aux_params, index=1):
        arg_params['query_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'query_' + str(index) + '_weight'])
        arg_params['query_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['query_' + str(index) + '_bias'])
        arg_params['key_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'key_' + str(index) + '_weight'])
        arg_params['key_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['key_' + str(index) + '_bias'])

        if cfg.RELATION.conv_z:
            arg_params['linear_out_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
                'linear_out_' + str(index) + '_weight'])
            arg_params['linear_out_' + str(index) + '_bias'] = mx.nd.zeros(
                shape=self.arg_shape_dict['linear_out_' + str(index) + '_bias'])

        if cfg.RELATION.conv_g:
            arg_params['value_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
                'value_' + str(index) + '_weight'])
            arg_params['value_' + str(index) + '_bias'] = mx.nd.zeros(
                shape=self.arg_shape_dict['value_' + str(index) + '_bias'])

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])
        for idx in range(2):
            self.init_weight_selsa(cfg, arg_params, aux_params, index=idx+1)

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rcnn(cfg, arg_params, aux_params)
