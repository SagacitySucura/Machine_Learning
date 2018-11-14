import os 
import tensorflow as tf 
import tensorflow.contrib as contrib
import numpy as np
import mnist_inference
import linecache
rootdir="H:/deep_learning/stripe_surface/deep_learning/Covnet/data/"
os.chdir(rootdir)

Batch_Size = 100
Learning_Rate_Base = 0.8
Learning_Rate_Decay = 0.99
Regularation_Rate = 0.0001
Train_steps = 30000
Moving_Average_Decay = 0.99
Model_Save_Path = 'path/'
Model_Name = 'model.ckpt'
train_size = len(os.listdir("trainColloction/"))

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
def train():
    x = tf.placeholder(
        tf.float32, [
        Batch_Size,
        mnist_inference.Image_Size,
        mnist_inference.Image_Size + 2,
        mnist_inference.Num_Channels],
        name='x_input'
    )
    y_ = tf.placeholder(
        tf.float32, [Batch_Size, mnist_inference.Output_Node],
        name='y_input'
    )
    regularizer = contrib.layers.l2_regularizer(Regularation_Rate)
    y = mnist_inference.inference(x, True, regularizer)

    varible_average = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
    varible_average_op = varible_average.apply(tf.trainable_variables())

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(
        y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]
    ))
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(1e-5)\
                 .minimize(loss)
    
    train_op = tf.group(train_step, varible_average_op)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(Train_steps):
            index = i%train_size
            label,data=readData('trainColloction/train'+str(index)+'.txt',mnist_inference.Output_Node+1,588)
# 这里只能用np.reshape()而不能用tf.reshape(),因为前者操作的是元组，后者操作的是张量，
# 而我们这里需要的就是元组而不是张量
            data = np.reshape(data, [
                Batch_Size,
                mnist_inference.Image_Size,
                mnist_inference.Image_Size + 2,
                mnist_inference.Num_Channels
            ])
            _, loss_value = sess.run([train_op, loss],
            feed_dict={x: data, y_: label[:,0:-1]})
            if(i % 1000 ==0):
                print("After %d training steps, loss on training "
                "batch is %g"%(i, loss_value))
                saver.save(
                    sess,os.path.join(Model_Save_Path, Model_Name)
                )
def main(argv=None):
    train()
if __name__ =='__main__':
    tf.app.run()