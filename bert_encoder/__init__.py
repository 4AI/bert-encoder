# -*- coding: utf-8 -*-

import os
import random
import numpy as np

from pprint import pprint
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from bert_encoder.bert.extract_features import convert_lst_to_features
from bert_encoder.bert.tokenization import FullTokenizer

__all__ = ['__version__', 'BertEncoder']
__version__ = '0.1.1'

from .helper import set_logger, import_tf
from .graph import optimize_bert_graph, PoolingStrategy

class BaseConfig(object):
    ckpt_name = "bert_model.ckpt"
    config_name = "bert_config.json"
    max_seq_len = 128
    pooling_layer = [-2]
    pooling_strategy = PoolingStrategy.REDUCE_MEAN
    mask_cls_sep = False


class BertEncoder(object):
    def __init__(self, model_dir, config=None, device_id=-1, verbose=False):
        """Intialize
        Args:
           - model_dir: dirname to pretrain model
           - config: instance of class extends BaseConfig
           - device_id: set computing unit: 0 specify CPU, 1 specify GPU
           - verbose: whether to show encoding log
        """
        if config is None:
            self.config = BaseConfig()
        else:
            self.config = config
        self.config.model_dir = model_dir
        self.config.verbose = verbose
        self.device_id = device_id
        self.mask_cls_sep = self.config.mask_cls_sep
        self.verbose = self.config.verbose
        self.graph_path = optimize_bert_graph(self.config)
        self.logger = set_logger("BertEncoder", self.config.verbose)

        self.tf = import_tf(self.device_id, self.verbose)
        self.estimator = self.get_estimator(self.tf)

    def get_estimator(self, tf):
        """Get tf estimator
        """
        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                'encodes': output[0]
            })

        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

    def encode(self, sentences):
        """Encode text to vector
        Args:
          - sentences: list of string
        """
        if self.verbose == 1:
            self.logger.info('use device %s, load graph from %s' %
                    ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))

        res = []
        for sentence in sentences:
            for r in self.estimator.predict(input_fn=self.input_fn_builder(self.tf, sentence),
                                            yield_single_examples=False):
                res.append(r['encodes'])
        return res

    def input_fn_builder(self, tf, sentence):        
        """Input function builder to estimator
        """

        def gen():
            tokenizer = FullTokenizer(vocab_file=os.path.join(self.config.model_dir, 'vocab.txt'))
            # check if msg is a list of list, if yes consider the input is already tokenized
            is_tokenized = all(isinstance(el, list) for el in sentence)
            tmp_f = list(convert_lst_to_features(sentence, self.config.max_seq_len, tokenizer, self.logger,
                                                    is_tokenized, self.mask_cls_sep))
            #print([f.input_ids for f in tmp_f])
            yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
            }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32},
                output_shapes={
                    'input_ids': (None, self.config.max_seq_len),
                    'input_mask': (None, self.config.max_seq_len),
                    'input_type_ids': (None, self.config.max_seq_len)}))

        return input_fn
