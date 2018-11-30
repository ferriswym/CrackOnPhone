#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:13:33 2018

@author: YumingWu
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from IPython.display import display

def test_tflite(model, image):
    
    interpreter = tf.contrib.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # test the model with given inputs
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    return output

if __name__ == "__main__":
    file = "../../data/CFD/selected/train/1.jpg"
    model = '../../tflite/model.tflite'
    image_temp = Image.open(file)
    height = image_temp.height
    width = image_temp.width
    image_temp = (np.array(Image.open(file), dtype=np.float32) - 127.5) / 127.5
    image = np.expand_dims(image_temp, axis=0)
    res = np.reshape(test_tflite(model, image), [height, width])
    display(Image.fromarray(res * 255).convert("L"))