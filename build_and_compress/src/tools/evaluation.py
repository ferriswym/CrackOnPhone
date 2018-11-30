#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 20:58:55 2018

@author: YumingWu
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from IPython.display import display

def eval_on_single_image(output, gt, height, width):
    TP = 0
    for x in range(height):
        for y in range(width):
            if output[x,y] == 1 and gt[x,y] == 1:
                TP += 1
    pr = TP / np.sum(output)
    re = TP / np.sum(gt)
    F1 = 2 * pr * re / (pr + re)
    return pr, re, F1

def preprocess(image_file):
    image = Image.open(image_file)
    height, width = image.height, image.width
    image = np.expand_dims((np.array(image, dtype=np.float32) - 127.5) / 127.5, axis=0)
    return image, height, width

def postprocess(output, height, width):
    output = output > 0.5
    output = np.multiply(output, 1)
    output = np.reshape(output, [height, width])
    return output
    
def evaluate_pb(pb_model, net_input_name, net_output_name, img_dir, gt_dir):
    
    Isdisplay = True
    
    ## load pb model
    with tf.Graph().as_default() as graph:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  
        sess = tf.Session(config=config)
        
        # restore the model
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_model, 'rb') as i_file:
            graph_def.ParseFromString(i_file.read())
        tf.import_graph_def(graph_def, name='import')
    
    # obtain input & output nodes and then test the model
    net_input = graph.get_tensor_by_name('import/' + net_input_name + ':0')
    net_output = graph.get_tensor_by_name('import/' + net_output_name + ':0')
    
    ## inference and evaluate on each image
    pr_sum = re_sum = F1_sum = 0
    images_list = os.listdir(img_dir)
    for file in images_list:
        image, height, width = preprocess(os.path.join(img_dir, file))
        output = sess.run(net_output, feed_dict={net_input: image})
        output = postprocess(output, height, width)
        gt = np.array(Image.open(os.path.join(gt_dir, file[:-3]) + 'png')) // 255
        if Isdisplay:
            dis = Image.fromarray((output*255).astype(np.uint8))
            dis_gt = Image.open(os.path.join(gt_dir, file[:-3]) + 'png')
            display(dis)
            display(dis_gt)
        Pr, Re, F1 = eval_on_single_image(output, gt, height, width)
        print("%s: precision=%.4f, recall=%.4f, F1=%.4f"%(file, Pr, Re, F1))
        pr_sum += Pr
        re_sum += Re
        F1_sum += F1
    num = len(images_list)
    print("avg_pr:%.4f, avg_re:%.4f, avg_F1:%.4f"%(pr_sum/num, re_sum/num, F1_sum/num))
        
def evaluate_tflite(model, img_dir, gt_dir):
    interpreter = tf.contrib.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # test the model on each image
    pr_sum = re_sum = F1_sum = 0
    images_list = os.listdir(img_dir)
    for file in images_list:
        image, height, width = preprocess(os.path.join(img_dir, file))
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = postprocess(output, height, width)
        gt = np.array(Image.open(os.path.join(gt_dir, file[:-3]) + 'png')) // 255
        Pr, Re, F1 = eval_on_single_image(output, gt, height, width)
        print("%s: precision=%.4f, recall=%.4f, F1=%.4f"%(file, Pr, Re, F1))
        pr_sum += Pr
        re_sum += Re
        F1_sum += F1
    num = len(images_list)
    print("avg_pr:%.4f, avg_re:%.4f, avg_F1:%.4f"%(pr_sum/num, re_sum/num, F1_sum/num))

if __name__ == '__main__':
    pb_model = '../../models/frozen.pb'
#    pb_model = "../../models/pruned_model.ckpt"
    tflite = '../../tflite/model.tflite'
    img_dir = '../../data/6/image'
    gt_dir = '../../data/6/gt'
    input_name = 'images'
    output_name = 'output'
    evaluate_pb(pb_model, input_name, output_name, img_dir, gt_dir)
#    evaluate_tflite(tflite, img_dir, gt_dir)