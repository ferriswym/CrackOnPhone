# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:30:30 2018

@author: wuyuming
"""

from tensorflow.python.platform import gfile
import tensorflow as tf

model = '../../models/frozen.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('../../log/', graph)
