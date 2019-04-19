import tensorflow as tf
import scipy.io
import numpy as np
from time import strftime, localtime
np.random.seed(1)

batch_size = 64
lr = 0.0001

def next_batch(l, batch_size):
    index = np.arange(l)
    return np.random.permutation(index)[:batch_size]

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #strides[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def print_confusion_matrix(pre, true):
    confusion_matrix = [ [0 for j in range(4)] for i in range(4) ]
    for i in range(pre.shape[0]):
        confusion_matrix[true[i][0]][pre[i][0]] += 1
    print('\nConfusion Matrix:')
    for i in range(4):
        print(confusion_matrix[i])
    print('\n')

def main():
    data = scipy.io.loadmat('data.mat')
    x_train = data['train_de']
    y_train_raw = data['train_label_eeg']
    x_test = data['test_de']
    y_test_raw = data['test_label_eeg']
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    y_train = np.zeros([y_train_raw.shape[0], 4])
    y_test = np.zeros([y_test_raw.shape[0], 4])
    for i in range(y_train_raw.shape[0]):
        y_train[i][y_train_raw[i][0]] = 1
    for i in range(y_test_raw.shape[0]):
        y_test[i][y_test_raw[i][0]] = 1

    xs = tf.placeholder(tf.float32, [None, 310])
    ys = tf.placeholder(tf.float32, [None, 4])
    keep_prob = tf.placeholder(tf.float32) # rate = 1 - keep_prob

    xs_exchange = tf.reshape(xs,[-1,31,10,1]) # size: 31*10*1

    #conv1 layer
    w_conv1 = weight_variable([5,5,1,16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(xs_exchange,w_conv1)+b_conv1) # output size: 31*10*16
    h_pool1 = max_pool_2x2(h_conv1) # output size: 16*5*16

    #conv2 layer
    w_conv2 = weight_variable([5,5,16,32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2) # output size: 16*5*32
    h_pool2 = max_pool_2x2(h_conv2) # output size: 8*3*32

    #fc1 layer
    h_pool2_flat = tf.reshape(h_pool2,[-1,8*3*32])
    W_fc1 = weight_variable([8*3*32,250])
    b_fc1 = bias_variable([250])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #fc2 layer
    W_fc2 = weight_variable([250,4])
    b_fc2 = bias_variable([4])
    output = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    sigmoid_output = tf.nn.sigmoid(output)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=ys))
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    print(strftime("%H:%M:%S", localtime()), "start")
    for it in range(25000):
        rnd_ind = next_batch(x_train.shape[0], batch_size)
        # c,p = sess.run([h_conv2, h_pool2], feed_dict={xs: x_train[rnd_ind], ys: y_train[rnd_ind], rate:0.5})
        # print(c.shape, p.shape)
        _, train_loss = sess.run([train_step, loss], feed_dict={xs: x_train[rnd_ind], ys: y_train[rnd_ind], keep_prob:0.5})

        if it % 500 == 0:
            print("Iter: %d. Loss: %f" % (it, train_loss))

            pre_raw = sess.run(sigmoid_output, feed_dict={xs: x_test, keep_prob:1})
            pre = np.zeros([pre_raw.shape[0], 1], dtype=np.int64)
            for i in range(pre_raw.shape[0]):
                ind = np.argmax(pre_raw[i])
                pre[i][0] = ind
            
            acc = np.mean(pre == y_test_raw)
            print("Test accuracy =", acc, '\n')

    pre_raw = sess.run(sigmoid_output, feed_dict={xs: x_test, keep_prob:1})
    pre = np.zeros([pre_raw.shape[0], 1], dtype=np.int64)
    for i in range(pre_raw.shape[0]):
        ind = np.argmax(pre_raw[i])
        pre[i][0] = ind
    
    acc = np.mean(pre == y_test_raw)
    print(strftime("%H:%M:%S", localtime()), "Test accuracy =", acc)
    print_confusion_matrix(pre, y_test_raw)

if __name__ == '__main__':
    main()
