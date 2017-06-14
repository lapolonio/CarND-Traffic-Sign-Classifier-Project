"""
LeNet Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from keras.layers import Dense, Activation, Conv2D, Flatten

EPOCHS = 10
BATCH_SIZE = 50

layer_width = {
    'layer_1': 6,
    'layer_2': 16,
    'fully_connected_1': 120
}

# Store layers weight & bias
weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 1, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'fully_connected_1': tf.Variable(tf.truncated_normal(
        [5 * 5 * layer_width['layer_2'], layer_width['fully_connected_1']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected_1'], n_classes]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'fully_connected_1': tf.Variable(tf.zeros(layer_width['fully_connected_1'])),
    'out': tf.Variable(tf.zeros(n_classes))
}


def conv2d(x, W, b, strides=1, padding='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # TODO: Define the LeNet architecture.
    # Layer 1 - 32*32*1 to 14x14x6
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'], padding='VALID')
    conv1 = maxpool2d(conv1)

    # Layer 2 - 14x14x6 to 5*5*16
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'], padding='VALID')
    conv2 = maxpool2d(conv2)

    # Fully connected layer - 5*5*16 to 512
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.contrib.layers.flatten(conv2)

    fc1 = tf.matmul(fc1, weights['fully_connected_1']) + biases['fully_connected_1']
    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction - 512 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])


    # Return the result of the last fully connected layer.

    return out


# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (32, 32))
# Classify over n_classes
y = tf.placeholder(tf.float32, (None, n_classes))
fc2 = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples


if __name__ == '__main__':
    # Load data

    validation_size = n_train *.1
    #TODO: Need to implement a bit more complicated
    validation_images = X_train[:validation_size]
    validation_labels = y_train[:validation_size]
    train_images = X_train[validation_size:]
    train_labels = X_train[validation_size:]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(X_test, y_test)

    gtsd = base.Datasets(train=train, validation=validation, test=test)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = gtsd.train.num_examples // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = gtsd.train.next_batch(BATCH_SIZE)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(gtsd.validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))

        # Evaluate on the test data
        test_loss, test_acc = eval_data(gtsd.test)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))
