import tensorflow as tf
import scipy.io
import numpy as np
from time import strftime, localtime
from multiprocessing import Process, Array
np.random.seed(1)
h_dim = 512
batch_size = 64
lr = 0.0001
max_iter = 100000
step_iter = max_iter//10
block_num = 9
conb_num = 3

data = scipy.io.loadmat('data.mat')
x_train = data['train_de']
y_train_raw = data['train_label_eeg']
x_test = data['test_de']
y_test_raw = data['test_label_eeg']

def random_batch(l, batch_size):
    rnd_indices = np.random.randint(0, l, batch_size)
    return rnd_indices

def next_batch(l, blocks, idx):
    block_size = l // blocks
    if idx == blocks - 1:
        return np.array(range(idx * block_size, l))
    return np.array(range(idx * block_size, (idx+1) * block_size))

def p(label, result_raw):
    data = scipy.io.loadmat('data.mat')
    x_train = data['train_de']
    y_train_raw = data['train_label_eeg']
    x_test = data['test_de']
    y_test_raw = data['test_label_eeg']
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    y_train = np.zeros([y_train_raw.shape[0], 1])
    y_test = np.zeros([y_test_raw.shape[0], 1])
    for i in range(y_train_raw.shape[0]):
        if y_train_raw[i][0] == label:
            y_train[i][0] = 1
        else:
            y_train[i][0] = 0
    for i in range(y_test_raw.shape[0]):
        if y_test_raw[i][0] == label:
            y_test[i][0] = 1
        else:
            y_test[i][0] = 0

    xs = tf.placeholder(tf.float32, [None, 310])
    ys = tf.placeholder(tf.float32, [None, 1])

    w1 = tf.Variable(tf.random_normal([310, h_dim]))
    b1 = tf.Variable(tf.zeros([1, h_dim]) + 0.1)
    w2 = tf.Variable(tf.random_normal([h_dim, 1]))
    b2 = tf.Variable(tf.zeros([1, 1]) + 0.1)

    h = tf.nn.sigmoid(tf.matmul(xs, w1) + b1)
    output = tf.matmul(h, w2) + b2
    sigmoid_output = tf.nn.sigmoid(output)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=ys))
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    print(strftime("%H:%M:%S", localtime()), "p%d start" % (label))
    pre_all = []
    for block in range(block_num):
        sess.run(tf.global_variables_initializer())
        block_id = random_batch(x_train.shape[0], x_train.shape[0] // block_num)
        x_block = x_train[block_id]
        y_block = y_train[block_id]
        for it in range(max_iter):
            rnd_ind = random_batch(x_block.shape[0], batch_size)
            _, train_loss = sess.run([train_step, loss], feed_dict={xs: x_block[rnd_ind], ys: y_block[rnd_ind]})

            if it % step_iter == 0:
                print("p%d Block: %d Iter: %d. Loss: %f" % (label, block, it, train_loss))
                pre_raw = sess.run(sigmoid_output, feed_dict={xs: x_block})
                pre = np.zeros([pre_raw.shape[0], 1])
                for i in range(pre_raw.shape[0]):
                    if pre_raw[i][0] >= 0.5:
                        pre[i][0] = 1
                    else:
                        pre[i][0] = 0
                
                acc = np.mean(pre == y_block)
                print("Train accuracy =", acc, '\n')

        pre_raw = sess.run(sigmoid_output, feed_dict={xs: x_test})
        pre = np.zeros([pre_raw.shape[0], 1])
        for i in range(pre_raw.shape[0]):
            #result_raw[label][i] = pre_raw[i][0]
            if pre_raw[i][0] >= 0.5:
                pre[i][0] = 1
            else:
                pre[i][0] = 0
    
        acc = np.mean(pre == y_test)
        pre_all.append(pre_raw)
        print("p%d Block: %d Test accuracy =" % (label, block), acc)
    
    pre_all = np.array(pre_all)
    bs = block_num // conb_num;
    for idx in range(y_test.shape[0]):
        max_ls = []
        for i in range(conb_num):
            max_ls.append(min(pre_all[j][idx][0] for j in range(i*bs, (i+1)*bs)))
        result_raw[label][idx] = max(max_ls)

    pre = np.zeros([y_test.shape[0], 1])
    for i in range(y_test.shape[0]):
        if result_raw[label][i] >= 0.5:
            pre[i][0] = 1
        else:
            pre[i][0] = 0
    acc = np.mean(pre == y_test)
    print(strftime("%H:%M:%S", localtime()), "p%d Test accuracy =" % (label), acc)



def main():
    result_raw = [Array('f', range(y_test_raw.shape[0])) for i in range(4)]

    P0 = Process(target=p, args=(0, result_raw))
    P1 = Process(target=p, args=(1, result_raw))
    P2 = Process(target=p, args=(2, result_raw))
    P3 = Process(target=p, args=(3, result_raw))
    P0.start()
    P1.start()
    P2.start()
    P3.start()
    P0.join()
    P1.join()
    P2.join()
    P3.join()
    result = np.zeros([y_test_raw.shape[0], 1])
    for i in range(y_test_raw.shape[0]):
        mm = -1
        ind = -1
        for j in range(4):
            if result_raw[j][i] > mm:
                mm = result_raw[j][i]
                ind = j
        result[i][0] = ind

    acc = np.mean(result == y_test_raw)
    print("Total accuracy:", acc)

if __name__ == '__main__':
    main()