from abc import ABCMeta
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset

class Network():
    __metaclass__ = ABCMeta

    trainig_data = None
    test_data = None
    validation = None
    ops_loss_train = None
    ops_acc_train = None
    ops_min_step = None

    def network(self, x):
        pass

    def init_io_handle(self):
        self.io_handle = tf.train.Saver(tf.get_collection(tf.trainable_variables()))

    def save(self, sess, path):
        self.io_handle.save(sess, path)

    def restore(self, sess, path):
        if self.io_handle is not None:
            self.io_handle.restore(sess, path)

    def loss(self, output, labels):
        pass

    def accuracy(self, output, batch_labels):
        pass

    def build(self):
        pass

class Mnist(Network):

    def __init__(self):
        def get_data(data, mode='train'):
            mode_data = getattr(data, mode)
            images = tf.constant(mode_data.images, dtype=tf.float32, name="MNIST_IMAGES_" + mode)
            images = tf.reshape(images, [-1, 28, 28, 1])
            labels = tf.constant(mode_data.labels, dtype=tf.int64, name="MNIST_LABELS_" + mode)
            return images, labels
        data = mnist_dataset.load_mnist()
        self.training_data, self.test_data, self.validation_data = dict(), dict(), dict()
        self.training_data['images'], self.training_data['labels'] = get_data(data, 'train')
        self.test_data['images'], self.test_data['labels'] = get_data(data, 'test')
        self.validation_data['images'], self.validation_data['labels'] = get_data(data, 'validation')


    @staticmethod
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

class Mnist_N0(Mnist):


    def network(self, x):
        """deepnn builds the graph for a deep net for classifying digits.
          Args:
            x: an input tensor with the dimensions (N_examples, 784), where 784 is the
            number of pixels in a standard MNIST image.
          Returns:
            A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
            equal to the logits of classifying the digit into one of 10 classes (the
            digits 0-9). keep_prob is a scalar placeholder for the probability of
            dropout.
          """
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 8 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = Mnist.weight_variable([5, 5, 1, 8])
            b_conv1 = Mnist.bias_variable([8])
            h_conv1 = tf.nn.relu(Mnist.conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = Mnist.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 8 feature maps to 16.
        with tf.name_scope('conv2'):
            W_conv2 = Mnist.weight_variable([5, 5, 8, 16])
            b_conv2 = Mnist.bias_variable([16])
            h_conv2 = tf.nn.relu(Mnist.conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = Mnist.max_pool_2x2(h_conv2)

        # Second convolutional layer -- maps 16 feature maps to 32.
        with tf.name_scope('conv3'):
            W_conv3 = Mnist.weight_variable([5, 5, 16, 32])
            b_conv3 = Mnist.bias_variable([32])
            h_conv3 = tf.nn.relu(Mnist.conv2d(h_pool2, W_conv3) + b_conv3)

        # Third pooling layer.
        with tf.name_scope('pool3'):
            h_pool3 = Mnist.max_pool_2x2(h_conv3)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x16 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = Mnist.weight_variable([512, 1024])
            b_fc1 = Mnist.bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool3, [-1, 512])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = Mnist.weight_variable([1024, 10])
            b_fc2 = Mnist.bias_variable([10])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return y_conv, keep_prob

    def get_batch(self, mode='train'):
        data_holder = None
        if mode == 'train':
            data_holder = self.training_data
        elif mode == 'validation':
            data_holder = self.validation_data
        elif mode == 'test':
            data_holder = self.test_data
        indices = tf.random_uniform([128], 0, data_holder['images'].get_shape()[0].value, tf.int64)
        batch_images = tf.gather(data_holder['images'], indices)
        batch_labels = tf.gather(data_holder['labels'], indices)
        batch_labels = tf.one_hot(batch_labels, 10, name="labels" + mode)
        return batch_images, batch_labels

    def loss(self, output, batch_labels):
        with tf.name_scope('loss'):
            batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=batch_labels)
        return tf.reduce_mean(batch_loss)

    def accuracy(self, output, batch_labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(batch_labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        return accuracy

    def build(self):
        batch_images, batch_labels = self.get_batch('train')
        output, _ = self.network(batch_images)
        self.ops_loss_train = self.loss(output, batch_labels)
        self.ops_acc = self.accuracy(output, batch_labels)
        self.ops_min_step = tf.train.AdamOptimizer(1e-4).minimize(self.ops_loss_train)
