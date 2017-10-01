import networks
import tensorflow as tf

n0 = networks.Mnist_N0()
n0.build()

sess = tf.InteractiveSession()