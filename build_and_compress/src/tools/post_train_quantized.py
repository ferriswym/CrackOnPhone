#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:22:57 2018

@author: YumingWu
"""

import tensorflow as tf

graph_def_file = '../../models/frozen.pb'
input_arrays = ["images"]
output_arrays = ["output/logits"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)
converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.post_training_quantize = True
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}
converter.default_ranges_stats = (0., 6.)
tflite_model = converter.convert()
open("../tflite/quantized_model.tflite", "wb").write(tflite_model)
print('successfully convert quantized pb to tflite')