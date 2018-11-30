#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:20:49 2018

@author: YumingWu
"""

import tensorflow as tf

graph_def_file = "../pretrained/mobilenet_v1_1.0_224_quant/mobilenet_v1_1.0_224_quant_frozen.pb"
input_arrays = ["input"]
output_arrays = ["MobilenetV1/Predictions/Reshape_1"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)
converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}
tflite_model = converter.convert()
open("../tflite/q_quantized_model.tflite", "wb").write(tflite_model)