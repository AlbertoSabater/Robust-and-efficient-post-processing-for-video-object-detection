# from rcnn.config import config
from mxnet.symbol import BatchNorm as BN

bn_count = [0]

def BatchNorm(config, **kwargs):
    if config.SYNC_BN:
        from mxnet.symbol.contrib import SyncBatchNorm
        bn_count[0] = bn_count[0] + 1
        ndev = len(config.gpus.split(','))
        # ndev = len(config.gpus)
        wd_mult = 1 if config.SYNC_BN_WD else 0
        # sym = SyncBatchNorm(data, fix_gamma=False, use_global_stats=False, eps=config.BN_EPS, wd_mult=wd_mult, ndev=ndev, key=str(bn_count[0]), **kwargs)
        kwargs.pop('use_global_stats')
        # learn params
        print('ndev', ndev, 'bn_count', bn_count[0])
        sym = SyncBatchNorm(use_global_stats=False, wd_mult=wd_mult, ndev=ndev, key=str(bn_count[0]), **kwargs)
        # sym = SyncBatchNorm(wd_mult=wd_mult, ndev=ndev, key=str(bn_count[0]), **kwargs)
        print("{} using SyncBN".format(sym))
    else:
        # sym = BN(data, fix_gamma=False, use_global_stats=True, eps=config.BN_EPS, **kwargs)
        sym = BN(**kwargs)
        # print("{} using BatchNorm".format(sym))
    return sym

