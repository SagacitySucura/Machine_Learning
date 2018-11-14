import tensorflow as tf
import os
import os.path
import numpy as np
import linecache
rootdir="H:/deep_learning/stripe_surface/deep_learning/Covnet/data/"
os.chdir(rootdir)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def readData(filename,labellen,datalen):
    lines=linecache.getlines(filename)
    linecache.clearcache()
    new_label=[]
    new_data=[]
    for line in lines:
        new_line=[float(i) for i in line.split(',')]
        new_label.append(new_line[0:labellen])
        new_data.append(new_line[labellen:])
    return np.array(new_label),np.array(new_data)

train_size = len(os.listdir("trainColloction/"))
test_size = len(os.listdir("testColloction/"))
train_steps = 30000
Learning_Rate_Base = 0.8
Learning_Rate_Decay = 0.99
#定义 输入 占位符
x = tf.placeholder(tf.float32, [None,588])
y_ = tf.placeholder(tf.float32, [None,2])
W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,14,14,3])
h_conv1 = tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(h_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
# 2nd Conv Net
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                 strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu( h_conv2 + b_conv2 )
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
## Full connected layer
W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
## dropout layer
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_cov = tf.nn.softmax( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_cov,1e-10,1.0)),
                                reduction_indices=[1]))
global_step = tf.Variable(0, trainable=False)
Learning_Rate = tf.train.exponential_decay(
        Learning_Rate_Base,
        global_step =global_step,
        decay_steps = train_size,
        decay_rate = Learning_Rate_Decay
    )
train_step = tf.train.AdamOptimizer(Learning_Rate).minimize(cross_entropy,global_step=global_step)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(train_steps):
    index = i%train_size
    label,data=readData('trainColloction/train'+str(index)+'.txt',3,588)
    train_step.run({x:data,y_:label[:,0:-1],keep_prob:0.85})

for i in range(test_size):
    labelT,dataT=readData('testColloction/test'+str(i)+'.txt',2,588)
    yout=sess.run(y_cov,feed_dict={x:dataT,keep_prob:1.0})
    new_line=[]
    filehandle=open('result.txt','a',encoding='utf-8')
    for i in range(len(yout)):
        new_line.append(str(yout[i,0]))
        new_line.append(str(yout[i,1]))
        new_line.append(str(labelT[i,0]))
        filehandle.write(','.join(new_line))
        filehandle.write('\n')
        new_line=[]
    filehandle.close()