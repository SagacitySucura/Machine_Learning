import tensorflow as tf
import numpy as np
Input_Node = 240
Output_Node = 3
# 基础参数
Layer1_Node = 500
Image_Size = 8
Num_Channels = 3
Num_Labels = 3
# 第一层卷积层数据
Conv1_Size = 5
Conv1_Deep = 32
# 第二层卷积层数据
Conv2_Size = 5
Conv2_Deep = 64
# 全链接层
FC_SIze = 512

def get_weight_varible(shape, regularizer):
    weight = tf.get_variable(
        "weight", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.01) 
    )
    if(regularizer is not None):
        tf.add_to_collection(
            "losses",
            regularizer(weight)
        )
    return weight

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1_Conv1'):
        kernel = tf.get_variable(
            'kernel', [Conv1_Size, Conv1_Size, Num_Channels, Conv1_Deep],
            initializer=tf.truncated_normal_initializer(stddev=0.01)
        )
        biases = tf.get_variable(
            'biases', [Conv1_Deep], initializer=tf.constant_initializer(0)
        )
        conv1 = tf.nn.conv2d(
            input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
    with tf.variable_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )
    with tf.variable_scope('layer3_conv2'):
        kernel = tf.get_variable(
            'kernel', [Conv2_Size, Conv2_Size, Conv1_Deep, Conv2_Deep],
            initializer=tf.truncated_normal_initializer(stddev=0.01)
        )
        biases = tf.get_variable(
            'biases', [Conv2_Deep], initializer=tf.constant_initializer(0)
        )
        conv2 = tf.nn.conv2d(
            pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
    with tf.variable_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )
# 得到的pool2的维度为[batch_size*7*7*64]
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5_fc1'):
        weight = get_weight_varible([nodes, FC_SIze], regularizer)
        biases = tf.get_variable(
            'biases', [FC_SIze], initializer=tf.constant_initializer(0.0)
        )
        fc1 = tf.nn.relu(
            tf.matmul(reshaped, weight) + biases
        )
        if(train):
            fc1 = tf.nn.dropout(fc1, 1.0)
    with tf.variable_scope('layer6_fc2'):
        weight = get_weight_varible([FC_SIze, Output_Node], regularizer)
        biases = tf.get_variable(
            'biases', [Output_Node], initializer=tf.constant_initializer(0.0)
        ) 
        fc2 = tf.nn.softmax(tf.matmul(fc1, weight) + biases)

    return fc2