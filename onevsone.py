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

def p(label0, label1, result_raw, pid):
    data = scipy.io.loadmat('data.mat')
    x_train = data['train_de']
    y_train_raw = data['train_label_eeg']
    x_test = data['test_de']
    y_test_raw = data['test_label_eeg']

    #y_train = np.zeros([y_train_raw.shape[0], 1])
    #y_test = np.zeros([y_test_raw.shape[0], 1])
    y_train = []
    y_test = []
    idx_train = []
    for i in range(y_train_raw.shape[0]):
        if y_train_raw[i][0] == label1:
            idx_train.append(i)
            y_train.append([1])
        elif y_train_raw[i][0] == label0:
            idx_train.append(i)
            y_train.append([0])
    idx_train = np.array(idx_train)
    y_train = np.array(y_train)
    idx_test = []
    for i in range(y_test_raw.shape[0]):
        if y_test_raw[i][0] == label1:
            idx_test.append(i)
            y_test.append([1])
        elif y_test_raw[i][0] == label0:
            idx_test.append(i)
            y_test.append([0])
    idx_test = np.array(idx_test)
    y_test = np.array(y_test)
    x_train = x_train[idx_train]
    x_test = x_test[idx_test]


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
    
    print(strftime("%H:%M:%S", localtime()), "p%d start" % (pid))
    pre_all = []
    order = np.array(range(x_train.shape[0]))
    np.random.shuffle(order)
    for block in range(block_num):
        sess.run(tf.global_variables_initializer())
        block_id = next_batch(x_train.shape[0], block_num, block)
        x_block = x_train[order[block_id]]
        y_block = y_train[order[block_id]]
        for it in range(max_iter):
            rnd_ind = random_batch(x_block.shape[0], batch_size)
            _, train_loss = sess.run([train_step, loss], feed_dict={xs: x_block[rnd_ind], ys: y_block[rnd_ind]})

            if it % step_iter == 0:
                print("p%d Block: %d Iter: %d. Loss: %f" % (pid, block, it, train_loss))
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
            if pre_raw[i][0] >= 0.5:
                pre[i][0] = 1
            else:
                pre[i][0] = 0
    
        acc = np.mean(pre == y_test)
        pre_all.append(pre_raw)
        print("p%d Block: %d Test accuracy =" % (pid, block), acc)
    
    local_result = np.zeros([4, y_test_raw.shape[0]], dtype=np.float32)
    pre_all = np.array(pre_all)
    bs = block_num // conb_num;
    for idx in range(y_test.shape[0]):
        max_ls = []
        for i in range(conb_num):
            max_ls.append(min(pre_all[j][idx][0] for j in range(i*bs, (i+1)*bs)))
        r = max(max_ls)
        result_raw[label1][idx_test[idx]] += r
        result_raw[label0][idx_test[idx]] += (1 - r)
        local_result[label1][idx_test[idx]] = r

    pre = np.zeros([y_test.shape[0], 1])
    for i in range(y_test.shape[0]):
        if local_result[label1][idx_test[i]] >= 0.5:
            pre[i][0] = 1
        else:
            pre[i][0] = 0
    acc = np.mean(pre == y_test)
    print(strftime("%H:%M:%S", localtime()), "p%d Test accuracy =" % (pid), acc)



def main():
    result_raw = [Array('f', range(y_test_raw.shape[0])) for i in range(4)]
    for i in range(4):
        for j in range(y_test_raw.shape[0]):
            result_raw[i][j] = 0.0

    P0 = Process(target=p, args=(0, 1, result_raw, 0))
    P1 = Process(target=p, args=(0, 2, result_raw, 1))
    P2 = Process(target=p, args=(0, 3, result_raw, 2))
    P3 = Process(target=p, args=(1, 2, result_raw, 3))
    P4 = Process(target=p, args=(1, 3, result_raw, 4))
    P5 = Process(target=p, args=(2, 3, result_raw, 5))
    P0.start()
    P1.start()
    P2.start()
    P3.start()
    P4.start()
    P5.start()
    P0.join()
    P1.join()
    P2.join()
    P3.join()
    P4.join()
    P5.join()
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