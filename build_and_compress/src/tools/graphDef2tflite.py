# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:24:50 2018

@author: wuyuming
"""

import tensorflow as tf

graph_def_file = "../pretrained/mobilenet_v1_1.0_224_quant/mobilenet_v1_1.0_224_quant_frozen.pb"
input_arrays = ["input"]
output_arrays = ["MobilenetV1/Predictions/Reshape_1"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("../tflite/q_converted_model.tflite", "wb").write(tflite_model)
