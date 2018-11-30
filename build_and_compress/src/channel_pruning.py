# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:09:13 2018

@author: wuyuming
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from model_wrapper import Model
from PIL import Image
from tensorflow.python.framework import graph_util
slim = tf.contrib.slim

# fit to the Lasso model
def Lasso_fit(alpha, x, y):
    solver = LassoLars(alpha=alpha, fit_intercept=False, max_iter=3000)
    solver.alpha = alpha
    solver.fit(x, y)
    idxs = solver.coef_ != 0.
    c_cal = sum(idxs)
    return idxs, c_cal

# use linear regression to recontruct Y from changed X
def featuremap_reconstruction(x, y):
    reg = LinearRegression(copy_X=False, fit_intercept=False)
    reg.fit(x, y)
    return reg.coef_

# select the most import features by Lasso regression
def feature_selection(X, Y, W2, c_new, tolerance=0.02):
    
    ## intialize input shape and output shape
    initial_alpha = 1e-4
    c_in = X.shape[-1]    
    X_reshape = np.transpose(np.transpose(X, (0, 3, 1, 2)).reshape(X.shape[0],
                             X.shape[-1], -1), (1, 0, 2))
    W_reshape = np.transpose(W2, (2, 0, 1, 3)).reshape(W2.shape[2], -1, W2.shape[-1])
    product = np.matmul(X_reshape, W_reshape).reshape((c_in, -1)).T
    Y_reshape = Y.reshape(-1)
    
    ## intialize boundary of c
    left = 0
    right = initial_alpha
    lbound = c_new# - tolerance * c_in / 2
    rbound = c_new# + tolerance * c_in / 2
    
    ## get the right boundary of alpha
    while True:
        _, c_cal = Lasso_fit(right, product, Y_reshape)
        if c_cal < c_new:
            break
        else:
            right *= 2
    
    ## solve when c_cal in a appropriate range
    alpha = initial_alpha
    while True:
        if lbound < 0:
            lbound = 1
        idxs, c_cal = Lasso_fit(alpha, product, Y_reshape)
        if lbound <= c_cal and c_cal <= rbound:
            break
        elif abs(left - right) <= right * 0.1:
            if lbound > 1:
                lbound -= 1
            if rbound < c_in:
                rbound += 1
            left /= 1.2
            right *= 1.2
        elif c_cal > rbound:
            left += (alpha - left) / 2
        else:
            right -= (right - alpha) / 2
        
        if alpha < 1e-10:
            break
        
        alpha = (left + right) / 2
    c_new = c_cal
    
    return idxs, c_new

# 
def preprocess(image_dir):
    images_list = []
    for path in image_dir:
        path_list = os.listdir(path + '/image')
        images_list += [path + '/image/' + i for i in path_list]
    images = []
    for i in range(len(images_list) // 4):
        images.append((np.array(Image.open(images_list[i])) - 127.5) / 127.5)
    images = np.array(images)
    return images

# prune the channels and return new weights    
def prune(w2, X, Y):
    c_new = X.shape[-1] // 2
    idxs, c_new = feature_selection(X, Y, w2, c_new)
    neww2 = featuremap_reconstruction(X[:, :, :, idxs].reshape((X.shape[0], -1)), Y)
    neww2 = neww2.reshape((-1, 3, 3, c_new))
    neww2 = np.transpose(neww2, (1, 2, 3, 0))
    return idxs, neww2

# extract patches of features to extract input of convolution
def extract_input_output(model, conv_list):
    conv_list = conv_list[:10][1::2]
    patches = []
    outputs = []
    for op in conv_list:
        def_ = model.get_conv_def(op)
        output_tmp = model.get_output_by_op(op)
        outputs.append(output_tmp)
        input_tmp = model.get_input_by_op(op)
        ksize, stride, padding = def_['ksizes'], def_['strides'], def_['padding']
        patch = tf.extract_image_patches(input_tmp, ksize, stride, 
                                 rates=[1, 1, 1, 1], padding=padding)
        patches.append(patch)
    return patches, outputs

# extract patches of X from samples and Y 
def extract_X_Y(patch, feature):
    x_sample = np.random.randint(0, patch.shape[1] - 1, 10)
    y_sample = np.random.randint(0, patch.shape[2] - 1, 10)
    X = patch[:, x_sample, y_sample, :].reshape((-1, patch.shape[-1]))
    X = X.reshape((X.shape[0], 3, 3, -1))
    Y = feature[:, x_sample, y_sample, :].reshape((-1, feature.shape[-1]))
    return X, Y

# model pruning process
def prune_model(path, checkpoint, image_path):
    graph = tf.Graph()
    with graph.as_default():
        meta_path = os.path.join(path, checkpoint) + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph,config=config) as sess:
        checkpoint_path = os.path.join(path, checkpoint)
        saver.restore(sess, checkpoint_path)
        model = Model(sess)
        conv_list = model.get_operations_by_type()
        vars_dict = {model.get_var_by_op(op).name: model.param_data(op) for op in conv_list}
        bias = {key.split("/")[0] + "/bias:0": 
                slim.get_variables_by_name(key.split("/")[0] + "/bias")[0].eval(sess) for key in vars_dict}
        vars_dict.update(bias)
        inputs, outputs = extract_input_output(model, conv_list)
        images = preprocess(image_path)
        input_tensor = model.get_input_by_op(conv_list[0])
        forfeas, feats = sess.run([inputs, outputs], feed_dict={input_tensor: images})
        for i in range(5):
            # get variables and values of variables
            conv_front = conv_list[2*i]
            conv_after = conv_list[2*i + 1]
            w1 = model.get_var_by_op(conv_front)
            w2 = model.get_var_by_op(conv_after)
            b1 = slim.get_variables_by_name(os.path.split(conv_front.name)[0] + '/bias')[0]
            w1_val = model.param_data(conv_front)
            w2_val = model.param_data(conv_after)
            b1_val = b1.eval(sess)
            
            X,Y = extract_X_Y(forfeas[i], feats[i])
            idxs, neww2 = prune(w2_val, X, Y)
            vars_dict[w1.name] = w1_val[:, :, :, idxs]
            vars_dict[w2.name] = neww2
            vars_dict[b1.name] = b1_val[idxs]
    print("successfully complete pruning")
    return vars_dict

# redefine the graph after pruning
def conv2d(x, scope, vars_dict):
    with tf.variable_scope(scope):
        w_val = vars_dict[scope + '/weights' + ':0']
        b_val = vars_dict[scope + '/bias' + ':0']
        W = tf.get_variable('weights', shape=w_val.shape,
                            initializer=tf.constant_initializer(w_val))
        b = tf.get_variable('bias', shape=b_val.shape, 
                            initializer=tf.constant_initializer(b_val))
        net = tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')
        net = tf.nn.bias_add(net, b)
        net = tf.nn.relu(net)
        print(scope, net.get_shape().as_list())
    return net

def pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

def deconv2d(x, scope, vars_dict):
    with tf.variable_scope(scope):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        w_val = vars_dict[scope + '/weights' + ':0']
        b_val = vars_dict[scope + '/bias' + ':0']
        W = tf.get_variable('weights', shape=w_val.shape,
                            initializer=tf.constant_initializer(w_val))
        b = tf.get_variable('bias', shape=b_val.shape, 
                            initializer=tf.constant_initializer(b_val))
        net = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2,2,1],
                                     padding='SAME')
        net = tf.nn.bias_add(net, b)
        net = tf.nn.relu(net)
        print(scope, net.get_shape().as_list())
    return net

def copy_concat(x1, x2):
    return tf.concat([x1, x2], axis=3)

def inference(inputs, vars_dict):
    
    # down conv
    base = 8
    conv1 = conv2d(inputs, 'conv1', vars_dict)
    conv2 = conv2d(conv1, 'conv2', vars_dict)
    pool1 = pool(conv2)
    conv3 = conv2d(pool1, 'conv3', vars_dict)
    conv4 = conv2d(conv3, 'conv4', vars_dict)
    pool2 = pool(conv4)
    conv5 = conv2d(pool2, 'conv5', vars_dict)
    conv6 = conv2d(conv5, 'conv6', vars_dict)
    
    # up conv
    deconv2 = deconv2d(conv6, 'deconv2', vars_dict)
    deconcat2 = copy_concat(deconv2, conv4)
    upconv4 = conv2d(deconcat2, 'upconv4', vars_dict)
    upconv3 = conv2d(upconv4, 'upconv3', vars_dict)
    deconv1 = deconv2d(upconv3, 'deconv1', vars_dict)
    deconcat1 = copy_concat(deconv1, conv2)
    upconv2 = conv2d(deconcat1, 'upconv2', vars_dict)
    upconv1 = conv2d(upconv2, 'upconv1', vars_dict)
    
    ## output
    with tf.variable_scope('logits'):
        w_val = vars_dict['logits/weights:0']
        b_val = vars_dict['logits/bias:0']
        W = tf.get_variable('weights', shape=[3, 3, base, 1],
                            initializer=tf.constant_initializer(w_val))
        b = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(b_val))
        net = tf.nn.conv2d(upconv1, W, [1,1,1,1], padding='SAME')
        net = tf.nn.bias_add(net, b, name='logits')
    return net

# finetune after pruning        
def finetune(image_dir, eval_dir, epochs, vars_dict):
    
    height = 320
    width = 480
    
    ## build graph
    graph = tf.Graph()
    with graph.as_default():
        inputs_s = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        label_s = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        if inputs_s.shape[1] > inputs_s.shape[2]:
            inputs = tf.image.resize_image_with_crop_or_pad(
                    tf.image.tf.image.rot90(inputs_s), height, width)
            label = tf.image.resize_image_with_crop_or_pad(
                    tf.image.tf.image.rot90(label_s), height, width)
        else:
            inputs = tf.image.resize_image_with_crop_or_pad(inputs_s, height, width)
            label = tf.image.resize_image_with_crop_or_pad(label_s, height, width)
        logits = inference(inputs, vars_dict)
        output = tf.nn.sigmoid(logits, name='output')
        
        ## loss
        loss = -tf.reduce_mean(0.7*label*tf.log(tf.clip_by_value(output,1e-10,1.0)) + 
                              0.3*(1-label)*tf.log(tf.clip_by_value((1-output),1e-10,1.0)))
        
        ## optimizer
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        tf.add_to_collection('images', inputs)
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

        saver.save(sess, "../models/pruned_model.ckpt")
        print("successfully complete pruning and finetune")
        
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
    image_path = ['../data/1', '../data/2', '../data/3', '../data/4',
                '../data/5']
    eval_path = '../data/6'
    model_path = "../models"
    checkpoint = "distilled_model.ckpt"
    vars_dict = prune_model(model_path, checkpoint, image_path)
    finetune(image_path, eval_path, 100, vars_dict)