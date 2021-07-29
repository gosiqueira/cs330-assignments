import random

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm

from load_data import DataGenerator


def loss_function(preds, labels):
    '''
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    '''
    #############################
    #### YOUR CODE GOES HERE ####
    loss = tf.keras.losses.categorical_crossentropy(y_true=labels[:,-1:,:,:],
                                                    y_pred=preds[:, -1:,:,:],
                                                    from_logits=True)
    return tf.reduce_sum(loss)
    #############################

class MANN(tf.keras.Model):
    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        '''
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        '''
        #############################
        #### YOUR CODE GOES HERE ####
        _, K, N, D = input_images.shape
        in_zero = input_labels - input_labels[:, -1:, :, :]
        # Zero last N example, corresponding to the K+1 sample
        # Note the '-1'+':,' so num of dimensions keeps equal
        input_concat = tf.concat([input_images, in_zero], 3)
        input_concat = tf.reshape(input_concat, [-1, K*N, N + 784])
        out = self.layer2(self.layer1(input_concat))
        out  = tf.reshape(out, (-1, K, N, N))
        #############################
        return out


def mann(num_samples, num_classes, n_steps=50000, meta_batch_size=4):
    ims = tf.placeholder(tf.float32, shape=(
        None, num_samples + 1, num_classes, 784))
    labels = tf.placeholder(tf.float32, shape=(
        None, num_samples + 1, num_classes, num_classes))

    data_generator = DataGenerator(
        num_classes, num_samples + 1)

    o = MANN(num_classes, num_samples + 1)
    out = o(ims, labels)

    loss = loss_function(out, labels)
    optim = tf.train.AdamOptimizer(0.001)
    optimizer_step = optim.minimize(loss)

    test_hist = []

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        with tqdm(range(1, n_steps + 1)) as pbar:
            for step in pbar:
                i, l = data_generator.sample_batch('train', meta_batch_size)
                feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
                _, ls = sess.run([optimizer_step, loss], feed)

                if step % 100 == 0:
                    i, l = data_generator.sample_batch('test', 100)
                    feed = {ims: i.astype(np.float32),
                            labels: l.astype(np.float32)}
                    pred, tls = sess.run([out, loss], feed)
                    pred = pred.reshape(
                        -1, num_samples + 1,
                        num_classes, num_classes)
                    pred = pred[:, -1, :, :].argmax(2)
                    l = l[:, -1, :, :].argmax(2)
                    test_acc = (1.0 * (pred == l)).mean()

                    pbar.set_postfix(
                        test_loss='{0:.6f}'.format(ls),
                        test_accuracy='{0:.03f}'.format(test_acc),
                    )

                    test_hist.append(test_acc)

    return test_hist

if __name__ == '__main__':
    hist = []
    hist.append(mann(1, 2))
    hist.append(mann(1, 3))
    hist.append(mann(1, 4))
    hist.append(mann(5, 4))

    for h in hist:
        plt.plot(np.arange(len(h)), h)

    plt.title('MANN Test Accuracy')
    plt.legend(['k=1 n=2', 'k=1 n=3', 'k=1 n=4', 'k=5 n=4'])
    plt.savefig('mann.jpg')
