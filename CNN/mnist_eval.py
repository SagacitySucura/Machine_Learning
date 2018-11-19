import tensorflow as tf 
import mnist_inference
import mnist_train
import numpy as np
import os
import linecache
import os.path
rootdir="H:/deep_learning/stripe_surface/deep_learning/Covnet/data/"
os.chdir(rootdir)
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

def evaluate():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32, [
                mnist_train.Batch_Size,
                mnist_inference.Image_Size,
                mnist_inference.Image_Size + 2,
                mnist_inference.Num_Channels],
            'x_input'
        )
        y_ = tf.placeholder(
            tf.float32,
            [mnist_train.Batch_Size,
             mnist_inference.Num_Labels],
            'y_input'
        )
        y = mnist_inference.inference(x, False, None)
        varible_average = tf.train.ExponentialMovingAverage(
            mnist_train.Moving_Average_Decay
        )
        varible_to_restore = varible_average.variables_to_restore()
        saver = tf.train.Saver(varible_to_restore)
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(mnist_train.Model_Save_Path,
            mnist_train.Model_Name))
            for i in range(test_size):
                labelT,dataT=readData('testColloction/test'+str(i)+'.txt',1,588)
                dataT = np.reshape(dataT,[
                    mnist_train.Batch_Size,
                    mnist_inference.Image_Size,
                    mnist_inference.Image_Size + 2,
                    mnist_inference.Num_Channels
                ])
                yout=sess.run(y,feed_dict={x:dataT})
                new_line=[]
                filehandle=open('result.txt','a',encoding='utf-8')
                for i in range(len(yout)):
                    new_line.append(str(yout[i,0])+' ')
                    new_line.append(str(yout[i,1])+' ')
                    new_line.append(str(yout[i,2])+' ')
                    new_line.append(str(labelT[i,0]))
                    filehandle.write(','.join(new_line))
                    filehandle.write('\n')
                    new_line=[]
                filehandle.close()
                
def main(argv=None):
    evaluate()

if __name__ == "__main__":
    tf.app.run()