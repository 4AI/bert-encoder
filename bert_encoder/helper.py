# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import uuid
import pickle

__all__ = ['set_logger', 'import_tf']

class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger

def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf
