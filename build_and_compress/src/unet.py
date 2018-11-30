#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:41:08 2018

@author: YumingWu
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from IPython.display import display
from tensorflow.python.framework import graph_util
from matplotlib import pyplot as plt

REGULAR = 0.0

# define the components of the network
def conv2d(x, scope, inshape, outshape):
    with tf.variable_scope(scope):
        W = tf.get_variable('weights', shape=[3, 3, inshape, outshape],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(REGULAR))
        b = tf.get_variable('bias', shape=[outshape], initializer=tf.constant_initializer(0.0))
        net = tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')
        net = tf.nn.bias_add(net, b)
        net = tf.nn.relu(net)
        print(scope, net.get_shape().as_list())
    return net

def pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

def deconv2d(x,scope,inshape,outshape):
    with tf.variable_scope(scope):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        W = tf.get_variable('weights', shape=[2, 2, outshape, inshape],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(REGULAR))
        b = tf.get_variable('bias', shape=[outshape], initializer=tf.constant_initializer(0.0))
        net = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2,2,1], padding='SAME')
        net = tf.nn.bias_add(net, b)
        net = tf.nn.relu(net)
        print(scope, net.get_shape().as_list())
    return net

def separable_conv2d(x, scope, inshape, outshape):
    with tf.variable_scope(scope):
        depth_W = tf.get_variable('depthwise_weights', shape=[3, 3, inshape, 1],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(REGULAR))
        point_W = tf.get_variable('pointwise_weights', shape=[1, 1, inshape, outshape],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(REGULAR))
        b = tf.get_variable('bias', shape=[outshape], initializer=tf.constant_initializer(0.0))
        net = tf.nn.separable_conv2d(x, depth_W, point_W, [1,1,1,1], padding='SAME')
        net = tf.nn.bias_add(net, b)
        net = tf.nn.relu(net)
        print(scope, net.get_shape().as_list())
    return net

def copy_concat(x1, x2):
    return tf.concat([x1, x2], axis=3)

def accuracy(predictions, labels):
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    TP = np.nansum(predictions.astype(np.uint8) & labels.astype(np.uint8))
    precision = TP / np.nansum(predictions)
    recall = TP / np.nansum(labels)
    F1 = 2*precision*recall / (precision + recall)
    return F1

# define the main part of unet graph
def inference(inputs):
    
    ## down conv
    base = 64
    conv1 = conv2d(inputs, 'conv1', 3, base)
    conv2 = conv2d(conv1, 'conv2', base, base)
    pool1 = pool(conv2)
    conv3 = conv2d(pool1, 'conv3', base, base*2)
    conv4 = conv2d(conv3, 'conv4', base*2, base*2)
    pool2 = pool(conv4)
    conv5 = conv2d(pool2, 'conv5', base*2, base*4)
    conv6 = conv2d(conv5, 'conv6', base*4, base*4)
    pool3 = pool(conv6)
    conv7 = conv2d(pool3, 'conv7', base*4, base*8)
    conv8 = conv2d(conv7, 'conv8', base*8, base*8)
    pool4 = pool(conv8)
    conv9 = conv2d(pool4, 'conv9', base*8, base*16)
    conv10 = conv2d(conv9, 'conv10', base*16, base*16)
    
    ## up conv
    deconv4 = deconv2d(conv10, 'deconv4', base*16, base*8)
    deconcat4 = copy_concat(deconv4, conv8)
    upconv8 = conv2d(deconcat4, 'upconv8', base*16, base*8)
    upconv7 = conv2d(upconv8, 'upconv7', base*8, base*8)
    deconv3 = deconv2d(upconv7, 'deconv3', base*8, base*4)
    deconcat3 = copy_concat(deconv3, conv6)
    upconv6 = conv2d(deconcat3, 'upconv6', base*8, base*4)
    upconv5 = conv2d(upconv6, 'upconv5', base*4, base*4)
    deconv2 = deconv2d(upconv5, 'deconv2', base*4, base*2)
    deconcat2 = copy_concat(deconv2, conv4)
    upconv4 = conv2d(deconcat2, 'upconv4', base*4, base*2)
    upconv3 = conv2d(upconv4, 'upconv3', base*2, base*2)
    deconv1 = deconv2d(upconv3, 'deconv1', base*2, base)
    deconcat1 = copy_concat(deconv1, conv2)
    upconv2 = conv2d(deconcat1, 'upconv2', base*2, base)
    upconv1 = conv2d(upconv2, 'upconv1', base, base)
    
    ## output
    with tf.variable_scope('logits'):
        W = tf.get_variable('weights', shape=[1, 1, base, 1],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(REGULAR))
        b = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.0))
        net = tf.nn.conv2d(upconv1, W, [1,1,1,1], padding='SAME')
        net = tf.nn.bias_add(net, b, name='logits')
    return net

# experimental structure for an optional graph
def inference_separable(inputs):
    
    ## down conv
    base = 8
    conv1 = separable_conv2d(inputs, 'conv1', 3, base)
    conv2 = separable_conv2d(conv1, 'conv2', base, base)
    pool1 = pool(conv2)
    conv3 = separable_conv2d(pool1, 'conv3', base, base*2)
    conv4 = separable_conv2d(conv3, 'conv4', base*2, base*2)
    pool2 = pool(conv4)
    conv5 = separable_conv2d(pool2, 'conv5', base*2, base*4)
    conv6 = separable_conv2d(conv5, 'conv6', base*4, base*4)
#    pool3 = pool(conv6)
#    conv7 = conv2d(pool3, 'conv7', base*4, base*8)
#    conv8 = conv2d(conv7, 'conv8', base*8, base*8)
#    pool4 = pool(conv8)
#    conv9 = conv2d(pool4, 'conv9', base*8, base*16)
#    conv10 = conv2d(conv9, 'conv10', base*16, base*16)
    
    # up conv
#    deconv4 = deconv2d(conv10, 'deconv4', base*16, base*8)
#    deconcat4 = copy_concat(deconv4, conv8)
#    upconv8 = conv2d(deconcat4, 'upconv8', base*16, base*8)
#    upconv7 = conv2d(upconv8, 'upconv7', base*8, base*8)
#    deconv3 = deconv2d(conv8, 'deconv3', base*8, base*4)
#    deconcat3 = copy_concat(deconv3, conv6)
#    upconv6 = conv2d(deconcat3, 'upconv6', base*8, base*4)
#    upconv5 = conv2d(upconv6, 'upconv5', base*4, base*4)
    deconv2 = deconv2d(conv6, 'deconv2', base*4, base*2)
    deconcat2 = copy_concat(deconv2, conv4)
    upconv4 = separable_conv2d(deconcat2, 'upconv4', base*4, base*2)
    upconv3 = separable_conv2d(upconv4, 'upconv3', base*2, base*2)
    deconv1 = deconv2d(upconv3, 'deconv1', base*2, base)
    deconcat1 = copy_concat(deconv1, conv2)
    upconv2 = separable_conv2d(deconcat1, 'upconv2', base*2, base)
    upconv1 = separable_conv2d(upconv2, 'upconv1', base, base)

#    deconv2 = deconv2d(conv6, 'deconv2', base*4, base*2)
#    deconcat2 = copy_concat(deconv2, conv4)
#    upconv4 = conv2d(deconcat2, 'upconv4', base*4, base*2)
#    upconv3 = conv2d(upconv4, 'upconv3', base*2, base*2)
#    deconv1 = deconv2d(upconv3, 'deconv1', base*2, base)
#    deconcat1 = copy_concat(deconv1, conv2)
#    upconv2 = conv2d(deconcat1, 'upconv2', base*2, base)
#    upconv1 = conv2d(upconv2, 'upconv1', base, base)
    
    ## output
    with tf.variable_scope('logits'):
        depth_W = tf.get_variable('depthwise_weights', shape=[3, 3, base, 1],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(REGULAR))
        point_W = tf.get_variable('pointwise_weights', shape=[1, 1, base, 1],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(REGULAR))
        b = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.0))
        net = tf.nn.separable_conv2d(upconv1, depth_W, point_W, [1,1,1,1], padding='SAME')
        net = tf.nn.bias_add(net, b, name='logits')
    return net

def train(image_dir, eval_dir, epochs, convert_to_pb=True, convert_to_tflite=False):
    
    height = 320
    width = 480
    
    ## build graph
    graph = tf.Graph()
    with graph.as_default():
        inputs_s = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='images')
        label_s = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        if inputs_s.shape[1] > inputs_s.shape[2]:
            inputs = tf.image.resize_image_with_crop_or_pad(
                    tf.image.tf.image.rot90(inputs_s), height, width)
            label = tf.image.resize_image_with_crop_or_pad(
                    tf.image.tf.image.rot90(label_s), height, width)
        else:
            inputs = tf.image.resize_image_with_crop_or_pad(inputs_s, height, width)
            label = tf.image.resize_image_with_crop_or_pad(label_s, height, width)
        logits = inference(inputs)
#        logits = inference_separable(inputs)
        output = tf.nn.sigmoid(logits, name='output')
        
        ## loss        
        tf.losses.add_loss(-tf.reduce_mean(0.7*label*tf.log(tf.clip_by_value(output,1e-10,1.0)) + 
                              0.3*(1-label)*tf.log(tf.clip_by_value((1-output),1e-10,1.0))))
        
        loss = tf.losses.get_total_loss()
        
        ## optimizer
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        tf.add_to_collection('images', inputs_s)
        tf.add_to_collection('logits', logits)
        tf.add_to_collection('output', output)
        writer = tf.summary.FileWriter('../log', graph)
        writer.close()
        saver = tf.train.Saver()

    ## shuffle images filenames
    eval_list = os.listdir(eval_dir + '/image')
    images_list = []
    gt_list = []
    for path in image_dir:
        path_list = os.listdir(path + '/image')
        images_list += [path + '/image/' + i for i in path_list]
        gt_list += [path + '/gt/' + i[:-3] + 'png' for i in path_list]
    images_list = np.array(images_list)
    gt_list = np.array(gt_list)
    permu = np.random.permutation(images_list.shape[0])
    images_shuffled = images_list[permu].tolist()
    gt_shuffled = gt_list[permu].tolist()
    
    ## training process
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True       
    with tf.Session(graph=graph, config=config) as sess:
        tf.global_variables_initializer().run()
        train_plot = []
        valid_plot = []
        print('Initialized')
        for step in range(epochs):
            loss_avg = 0
            for num in range(len(images_shuffled)):
                image_tmp = (np.array(Image.open(images_shuffled[num])) - 127.5) / 127.5
                gt_tmp = np.array(Image.open(gt_shuffled[num])) // 255
                image = np.expand_dims(image_tmp, axis=0)
                gt = np.expand_dims(np.expand_dims(gt_tmp, axis=0), axis=3)
                feed_dict = {inputs_s: image, label_s:gt}
                _, predict, l = sess.run(
                        [optimizer, output, loss], feed_dict=feed_dict)
                loss_avg += l
            loss_avg /= len(images_shuffled) 
            eimage = []
            egt = []
            for num in range(len(eval_list) // 2):
                eval_file = eval_dir + '/image/' + eval_list[num]
                eval_gt = eval_dir + '/gt/' + eval_list[num][:-3] + 'png'
                eimage_tmp = (np.array(Image.open(eval_file)) - 127.5) / 127.5
                egt_tmp = np.array(Image.open(eval_gt)) // 255
                eimage.append(eimage_tmp)
                egt.append(egt_tmp)
            eimage = np.array(eimage)
            egt = np.expand_dims(np.array(egt), axis=3)
            feed_dict = {inputs: eimage, label:egt}
            out, l_e = sess.run([output, loss], feed_dict=feed_dict)
            train_plot.append(loss_avg)
            valid_plot.append(l_e)
            print("step:%d, training loss:%.7f, evaluation loss:%.7f"%(step,
                                            loss_avg, l_e))
#            display(Image.fromarray(egt[1, :, :, 0] * 255).convert("L"))
#            display(Image.fromarray((out[1, :, :, 0] * 255).astype(np.uint8)).convert("L"))

        saver.save(sess, "../models/model-%d.ckpt"%epochs)
        
        # plot
        plotx = np.arange(2, epochs)
        plt.plot(plotx, train_plot[2:], color="#000000", linestyle='--', label="training")
        plt.plot(plotx, valid_plot[2:], color="#A60628", label="validation")
        plt.tick_params(labelsize="larger")
        plt.xlabel("Training iterations", fontsize="x-large")
        plt.ylabel("loss", fontsize="x-large")
        plt.title("Learning curve", fontsize="x-large")
        plt.legend(fontsize="larger")
        
        if convert_to_pb:
            input_graph_def = graph.as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(
                    sess,
                    input_graph_def,
                    ['output']
                    )
            with tf.gfile.GFile('../models/frozen.pb', 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("convert ckpt to pb")
                    
        if convert_to_tflite:
            converter = tf.contrib.lite.TocoConverter.from_session(sess, [inputs], [output])
            tflite_model = converter.convert()
            open("../tflite/test.tflite", "wb").write(tflite_model)
    

if __name__ == '__main__':
    data_dir = ['../data/1', '../data/2', '../data/3', '../data/4',
                '../data/5']
    eval_dir = '../data/6'
    epochs = 30
    train(data_dir, eval_dir, epochs, convert_to_tflite=False)