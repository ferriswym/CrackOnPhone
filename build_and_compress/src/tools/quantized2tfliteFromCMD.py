#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:53:20 2018

@author: YumingWu
"""
import subprocess

file_path_pb = "/home/YumingWu/Projects/top/pretrained/mobilenet_v1/mobilenet_v1_1.0_224_frozen.pb"
file_path_tflite = "/home/YumingWu/Projects/top/tflite/quantized_model.tflite"
input_arrays = "input"
output_arrays = "MobilenetV1/Predictions/Reshape_1"

arg_list = [
    '--graph_def_file ' + file_path_pb,
    '--output_file ' + file_path_tflite,
    '--input_arrays ' + input_arrays,
    '--output_arrays ' + output_arrays]
arg_list += [
      '--inference_type QUANTIZED_UINT8',
      '--mean_values 128',
      '--std_dev_values 127']
arg_list += [
    '--default_ranges_min 0',
    '--default_ranges_max 6']
cmd_str = ' '.join(['tflite_convert'] + arg_list)
subprocess.call(cmd_str, shell=True)