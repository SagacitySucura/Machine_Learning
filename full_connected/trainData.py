import tensorflow as tf
import numpy as np
import os
import linecache
rootdir="H:/deep_learning/stripe_surface/deep_learning/allconnectnet/coord_data/"
os.chdir(rootdir)

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

#截断正态分布生成随机权重和偏置
def weight_variable(shape, name1, alpha=0.0001):
  initial = tf.Variable(tf.truncated_normal(shape, stddev=0.01),name=name1)
  tf.add_to_collection(
      "losses",tf.contrib.layers.l2_regularizer(alpha)(initial)
  )
  return initial
def bias_variable(shape,name1):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name1)

train_steps = 30000
train_size = len(os.listdir("trainColloction/"))
test_size = len(os.listdir("testColloction/"))
batch_size = 100
Learning_rate_base = 0.8
Learning_rate_decay = 0.99
Regularization_rate = 0.0001

x = tf.placeholder(tf.float32, [None, 588])
keep_prob=tf.placeholder(tf.float32)

W01 = weight_variable([588, 100],'W01',Regularization_rate)
b01 = bias_variable([100],'b01')
hidden1=tf.nn.relu(tf.matmul(x, W01) + b01)


W02 = weight_variable([100, 100],'W02',Regularization_rate)
b02 = bias_variable([100],'b02')
hidden2=tf.nn.relu(tf.matmul(hidden1, W02) + b02)


W03 = weight_variable([100, 100],'W03',Regularization_rate)
b03 = bias_variable([100],'b03')
hidden3=tf.nn.relu(tf.matmul(hidden2, W03) + b03)

hidden3=hidden3+hidden1
W04 = weight_variable([100, 100],'W04',Regularization_rate)
b04 = bias_variable([100],'b04')
hidden4=tf.nn.relu(tf.matmul(hidden3, W04) + b04)

W05 = weight_variable([100, 100],'W05',Regularization_rate)
b05 = bias_variable([100],'b05')
hidden5=tf.nn.relu(tf.matmul(hidden4, W05) + b05)

W06 = weight_variable([100, 100],'W06',Regularization_rate)
b06 = bias_variable([100],'b06')
hidden6=tf.nn.relu(tf.matmul(hidden5, W06) + b06)


hidden6=hidden6+hidden3
hidden6=tf.nn.dropout(hidden6,keep_prob)
W07 = weight_variable([100, 100],'W07',Regularization_rate)
b07 = bias_variable([100],'b07')
hidden7=tf.nn.relu(tf.matmul(hidden6, W07) + b07)

W08 = weight_variable([100, 2],'W08',Regularization_rate)
b08 = bias_variable([2],'b08')
hidden8 = tf.matmul(hidden7, W08) + b08

y = tf.nn.softmax(hidden8)


y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)),
                                reduction_indices=[1]))
tf.add_to_collection("losses",cross_entropy)



loss = tf.add_n(tf.get_collection("losses"))


global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(
    Learning_rate_base,
    global_step,
    train_size,
    Learning_rate_decay)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_steps):
        index = int(i%train_size)
        label,data=readData('trainColloction/train'+str(index)+'.txt',3,588)
        train_step.run({x:data,y_:label[:,0:-1],keep_prob:0.85})
    probability=0.0
    for i in range(test_size):
        labelT,dataT=readData('testColloction/test'+str(i)+'.txt',3,588)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        probability=probability+accuracy.eval({x:dataT,y_:labelT[:,0:-1],keep_prob:1.0})
#        yout=sess.run(y,feed_dict={x:dataT,keep_prob:1.0})
#        new_line=[]
#        filehandle=open('result.txt','a',encoding='utf-8')
#        for i in range(len(yout)):
#            new_line.append(str(yout[i,0]))
#            new_line.append(str(yout[i,1]))
#            new_line.append(str(labelT[i,0]))
#            filehandle.write(','.join(new_line))
#            filehandle.write('\n')
#            new_line=[]
#        filehandle.close()
    print(probability/test_size)



'''
W10 = sess.run(W01)
b10 = sess.run(b01)
W20 = sess.run(W02)
b20 = sess.run(b02)
W30 = sess.run(W03)
b30 = sess.run(b03)
W40 = sess.run(W04)
b40 = sess.run(b04)
np.save('w10',W10)
np.save('b10',b10)
np.save('w20',W20)
np.save('b20',b20)
np.save('w30',W30)
np.save('b30',b30)
np.save('w40',W40)
np.save('b40',b40)
'''