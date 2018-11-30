#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:49:11 2018

@author: YumingWu
"""

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib import graph_editor

HEIGHT = 320
WIDTH = 480

# convert checkpoint to *.pb file
def convert_ckpt_to_pb(path, checkpoint, pb_path):
    """Convert ckpt files to a *.pb model.
    
    Args:
    * path: the folder of ckpt files.
    * checkpoint: the *.ckpt file.
    * pb_path: file path to the *.pb model
    """
    graph = tf.Graph()
    with graph.as_default():
        meta_path = os.path.join(path, checkpoint) + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        inputs = tf.get_collection('images')[0]
        image = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='images')
        graph_editor.reroute_ts(image, inputs)
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        checkpoint_path = os.path.join(path, checkpoint)
        saver.restore(sess, checkpoint_path)
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                ['output']
                )
        with tf.gfile.GFile(pb_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    print('successfully convert to pb')

# convert checkpoint to *.tflite file
def convert_ckpt_to_tflite(path, checkpoint, input_name, output_name, output_path):
    """Convert ckpt files to a *.pb model.
    
    Args:
    * path: the folder of ckpt files.
    * checkpoint: the *.ckpt file.
    * input_name: network's input node's name
    * output_name: network's output node's name
    * output_path: file path to the *.tflite model
    """
    graph = tf.Graph()
    with graph.as_default():
        meta_path = os.path.join(path, checkpoint) + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        checkpoint_path = os.path.join(path, checkpoint)
        saver.restore(sess, checkpoint_path)
        input_tensor = graph.get_tensor_by_name(input_name + ':0')
        output_tensor = graph.get_tensor_by_name(output_name + ':0')
        converter = tf.contrib.lite.TocoConverter.from_session(
                sess, [input_tensor], [output_tensor])
        tflite_model = converter.convert()
        with tf.gfile.GFile(output_path, 'wb') as f:
            f.write(tflite_model)
    print('successfully convert cpkt to tflite')

# convert *.pb to *.tflite     
def convert_pb_model_to_tflite(file_path_pb, file_path_tflite, net_input_name, net_output_name):
    """Convert *.pb model to a *.tflite model.

    Args:
    * file_path_pb: file path to the *.pb model
    * file_path_tflite: file path to the *.tflite model
    * net_input_name: network's input node's name
    * net_output_name: network's output node's name
    """

    with tf.Graph().as_default():
        converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
            file_path_pb, [net_input_name], [net_output_name])
        tflite_model = converter.convert()
        with tf.gfile.GFile(file_path_tflite, 'wb') as o_file:
            o_file.write(tflite_model)
    print('successfully convert pb to tflite')
        
       
if __name__ == '__main__':
    path = '../../models'
#    checkpoint = "distilled_model.ckpt"#'model-400.ckpt'
    checkpoint = "pruned_model.ckpt"
    pb_path = '../../models/frozen.pb'
    lite_path = '../../tflite/model_test.tflite'
    input_name = 'images'
    output_name = 'output'
    convert_ckpt_to_pb(path, checkpoint, pb_path)
#    convert_ckpt_to_tflite(path, checkpoint, input_name, output_name, lite_path)
    convert_pb_model_to_tflite(pb_path, lite_path, input_name, output_name)
    