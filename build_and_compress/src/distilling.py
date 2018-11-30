#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 08:56:32 2018

The defined distilling process from a full-precision unet.

@author: YumingWu
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from IPython.display import display
from tensorflow.python.framework import graph_util

TEMPER = 4.0
REGULAR = 0.0

# get data and label from a given directory
def get_data_label(image_dir):
    
    ## shuffle images filenames
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
    return images_shuffled, gt_shuffled

# compute the soft target from training data by softmax(the source algorithm from Hinton)    
def get_soft_target(path, checkpoint, images_shuffled):
    graph = tf.Graph()
    with graph.as_default():
        meta_path = os.path.join(path, checkpoint) + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        inputs = tf.get_collection('images')
        logits = tf.get_collection('logits')
        soft_logits = tf.nn.softmax(logits[0] / TEMPER, axis = 3)
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph,config=config) as sess:
        checkpoint_path = os.path.join(path, checkpoint)
        saver.restore(sess, checkpoint_path)
        
        soft_target = []
        for num in range(len(images_shuffled)):
            image_tmp = (np.array(Image.open(images_shuffled[num])) - 127.5) / 127.5
            image = np.expand_dims(image_tmp, axis=0)
            feed_dict = {inputs[0]: image}
            soft_target.append(sess.run(soft_logits, feed_dict=feed_dict))
    return soft_target

# compute the soft target from training data by sigmoid
def get_sigmoid_logits(path, checkpoint, images_shuffled):
    graph = tf.Graph()
    with graph.as_default():
        meta_path = os.path.join(path, checkpoint) + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        inputs = tf.get_collection('images')
        logits = tf.get_collection('logits')
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph,config=config) as sess:
        checkpoint_path = os.path.join(path, checkpoint)
        saver.restore(sess, checkpoint_path)
        
        soft_target = []
        for num in range(len(images_shuffled)):
            image_tmp = (np.array(Image.open(images_shuffled[num])) - 127.5) / 127.5
            image = np.expand_dims(image_tmp, axis=0)
            feed_dict = {inputs[0]: image}
            soft_target.append(sess.run(logits, feed_dict=feed_dict))
    return soft_target

# define the structure of a graph
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

def copy_concat(x1, x2):
    return tf.concat([x1, x2], axis=3)

def inference(inputs):
    
    ## down conv
    base = 8
    conv1 = conv2d(inputs, 'conv1', 3, base)
    conv2 = conv2d(conv1, 'conv2', base, base)
    pool1 = pool(conv2)
    conv3 = conv2d(pool1, 'conv3', base, base*2)
    conv4 = conv2d(conv3, 'conv4', base*2, base*2)
    pool2 = pool(conv4)
    conv5 = conv2d(pool2, 'conv5', base*2, base*4)
    conv6 = conv2d(conv5, 'conv6', base*4, base*4)
    
    # up conv
    deconv2 = deconv2d(conv6, 'deconv2', base*4, base*2)
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

# distilling process    
def distill(soft_target, images_shuffled, gt_shuffled, eval_dir, epochs):
    
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
        soft_label = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
        logits = inference(inputs)
#        output = tf.arg_max(tf.nn.softmax(logits, axis=3), 3, name='output')
        output = tf.nn.sigmoid(logits, name='output')
        
        ## loss        
#        softmax_loss = tf.losses.softmax_cross_entropy(
#                tf.one_hot(tf.reshape(label, shape=(-1, height, width)), 2), logits)
#        tf.losses.add_loss(softmax_loss)
        
        sigmoid_loss = -tf.reduce_mean(0.7*label*tf.log(tf.clip_by_value(output,1e-10,1.0)) + 
                              0.3*(1-label)*tf.log(tf.clip_by_value((1-output),1e-10,1.0)))
        tf.losses.add_loss(sigmoid_loss)
        
        alpha = 1.0
#        distill_loss = alpha * tf.losses.softmax_cross_entropy(soft_label, logits / TEMPER)
        distill_loss = alpha * tf.reduce_sum((1-label) * tf.maximum(0.0, logits - soft_label)) / tf.reduce_sum(1 - label)
#        distill_loss = alpha * tf.reduce_mean(label * tf.maximum(0.0, soft_label - logits) + 
#                                (1 - label) * tf.maximum(0.0, logits - soft_label))
        tf.losses.add_loss(distill_loss)
        
        loss = tf.losses.get_total_loss()
        
        ## optimizer
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
        
        tf.add_to_collection('images', inputs)
        tf.add_to_collection('output', output)
        saver = tf.train.Saver()
        
    ## training process
    eval_list = os.listdir(eval_dir + '/image')
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
                feed_dict = {inputs_s: image, label_s:gt, soft_label:soft_target[num]}
                _, predict, l = sess.run(
                        [optimizer, output, sigmoid_loss], feed_dict=feed_dict)
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
            out, l_e = sess.run([output, sigmoid_loss], feed_dict=feed_dict)
            train_plot.append(loss_avg)
            valid_plot.append(l_e)
            print("step:%d, training loss:%.7f, evaluation loss:%.7f"%(step,
                                            loss_avg, l_e))
#            display(Image.fromarray(egt[1, :, :, 0] * 255).convert("L"))
#            display(Image.fromarray((out[1, :, :] * 255).astype(np.uint8)).convert("L"))

        saver.save(sess, "../models/distilled_model.ckpt")
        print("successfully saved distilled model")
        
        convert_to_pb = True
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
        
if __name__ == '__main__':
    model_path = '../models'
    checkpoint = 'model-30.ckpt'
    image_path = ['../data/1', '../data/2', '../data/3', '../data/4',
                '../data/5']
    eval_path = '../data/6'
    epochs = 100
    images_shuffled, gt_shuffled = get_data_label(image_path)
    soft_target = get_soft_target(model_path, checkpoint, images_shuffled)
    distill(soft_target, images_shuffled, gt_shuffled, eval_path, epochs)