#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:41:51 2018

@author: YumingWu
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from IPython.display import display

def test_pb_model(file_path, net_input_name, net_output_name, net_input_data):
    """Test the *.pb model.

    Args:
    * file_path: file path to the *.pb model
    * net_input_name: network's input node's name
    * net_output_name: network's output node's name
    * net_input_data: network's input node's data
    """

    with tf.Graph().as_default() as graph:
        sess = tf.Session()
        
        # restore the model
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as i_file:
            graph_def.ParseFromString(i_file.read())
        tf.import_graph_def(graph_def, name='import')
    
    # obtain input & output nodes and then test the model
    net_input = graph.get_tensor_by_name('import/' + net_input_name + ':0')
    net_output = graph.get_tensor_by_name('import/' + net_output_name + ':0')
    output = sess.run(net_output, feed_dict={net_input: net_input_data})
        
    return output

if __name__ == '__main__':
    file_path = '../../models/frozen.pb'
    input_name = 'images'
    output_name = 'output/logits'
    file = "../../data/CFD/selected/train/1.jpg"
    image_temp = Image.open(file)
    height = image_temp.height
    width = image_temp.width
    image_temp = (np.array(Image.open(file), dtype=np.float32) - 127.5) / 127.5
    image = np.expand_dims(image_temp, axis=0)
    res = np.reshape(test_pb_model(file_path, input_name, output_name, image), [height, width])
    display(Image.fromarray(res * 255).convert("L"))