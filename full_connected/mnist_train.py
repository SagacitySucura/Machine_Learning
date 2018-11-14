import os 
import tensorflow as tf 
import mnist_inference
import numpy as np
import linecache
import tensorflow.contrib as contrib
rootdir="H:/deep_learning/stripe_surface/deep_learning/allconnectnet/coord_data/"
os.chdir(rootdir)

Batch_Size = 100
Learning_Rate_Base = 0.1
Learning_Rate_Decay = 0.99
Regularation_Rate = 0.0001
Train_steps = 30000
Moving_Average_Decay = 0.99
Model_Save_Path = 'path/'
Model_Name = 'model.ckpt'
train_size = len(os.listdir("trainColloction/"))
test_size = len(os.listdir("testColloction/"))


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
        tf.float32, [None, mnist_inference.Input_Node],
        name='x_input'
    )
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.Output_Node],
        name='y_input'
    )
    regularizer = contrib.layers.l2_regularizer(Regularation_Rate)
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    varible_average = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
    varible_average_op = varible_average.apply(tf.trainable_variables())

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(
        y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]
    ))
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))

    Learning_Rate = tf.train.exponential_decay(
        Learning_Rate_Base,
        global_step =global_step,
        decay_steps = train_size,
        decay_rate = Learning_Rate_Decay
    )
    train_step = tf.train.GradientDescentOptimizer(Learning_Rate)\
                 .minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, varible_average_op)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(100):
            index = int(i%train_size)
            label,data=readData('trainColloction/train'+str(index)+'.txt',3,588)
            _, loss_value, step = sess.run([train_op, loss, global_step],
            feed_dict={x: data, y_: label[:,0:-1]})
            if(i % 1000 ==0):
                print("After %d training steps, loss on training"
                "batch is %g"%(step, loss_value))
                saver.save(
                    sess,os.path.join(Model_Save_Path, Model_Name)
                )
        probability = 0.0
        for i in range(10):
            labelT,dataT=readData('testColloction/test'+str(i)+'.txt',3,588)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            probability=probability+accuracy.eval({x:dataT,y_:labelT[:,0:-1]})
        print(probability/test_size)
def main(argv=None):
    train()
if __name__ =='__main__':
    tf.app.run()