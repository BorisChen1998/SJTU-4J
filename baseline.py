import tensorflow as tf
import scipy.io
import numpy as np
from time import strftime, localtime
np.random.seed(1)
h_dim = 512
batch_size = 64
lr = 0.0001

def random_batch(l, batch_size):
    rnd_indices = np.random.randint(0, l, batch_size)
    return rnd_indices

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

    w1 = tf.Variable(tf.random_normal([310, h_dim]))
    b1 = tf.Variable(tf.zeros([1, h_dim]) + 0.1)
    w2 = tf.Variable(tf.random_normal([h_dim, 4]))
    b2 = tf.Variable(tf.zeros([1, 4]) + 0.1)

    h = tf.nn.sigmoid(tf.matmul(xs, w1) + b1)
    output = tf.matmul(h, w2) + b2

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=ys))
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    print(strftime("%H:%M:%S", localtime()), "start")
    for it in range(500000):
        rnd_ind = random_batch(x_train.shape[0], batch_size)
        _, train_loss = sess.run([train_step, loss], feed_dict={xs: x_train[rnd_ind], ys: y_train[rnd_ind]})

        if it % 5000 == 0:
            print("Iter: %d. Loss: %f" % (it, train_loss))
            pre_raw = sess.run(output, feed_dict={xs: x_train})
            pre = np.zeros([pre_raw.shape[0], 1])
            for i in range(pre_raw.shape[0]):
                ind = np.argmax(pre_raw[i])
                pre[i][0] = ind
            
            acc = np.mean(pre == y_train_raw)
            print("Train accuracy =", acc)

            pre_raw = sess.run(output, feed_dict={xs: x_test})
            pre = np.zeros([pre_raw.shape[0], 1])
            for i in range(pre_raw.shape[0]):
                ind = np.argmax(pre_raw[i])
                pre[i][0] = ind
            
            acc = np.mean(pre == y_test_raw)
            print("Test accuracy =", acc, '\n')

    pre_raw = sess.run(output, feed_dict={xs: x_test})
    pre = np.zeros([pre_raw.shape[0], 1])
    for i in range(pre_raw.shape[0]):
        ind = np.argmax(pre_raw[i])
        pre[i][0] = ind
    
    acc = np.mean(pre == y_test_raw)
    print(strftime("%H:%M:%S", localtime()), "Test accuracy =", acc)

if __name__ == '__main__':
    main()