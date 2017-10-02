import networks
import tensorflow as tf
import numpy as np
import tables

n0 = networks.Mnist_N0()
n0.build()

print_interval = 500
total_iteration = 800000

log_folder = 'log_folder/'
weights_history_dump_path = log_folder + 'mnist_train_data.h5'
loss_file_path = log_folder + 'log_loss'

with tf.Session() as sess:
    flat_weights_list = []
    sess.run(tf.global_variables_initializer())

    network_weights = tf.trainable_variables()
    flat_network_weights = [tf.reshape(network_weight, shape=[1, -1]) for network_weight in network_weights]
    concatenated_weights = tf.concat([flat_network_weights[0], flat_network_weights[1]], 1)
    for weights in flat_network_weights[2:]:
        concatenated_weights = tf.concat([concatenated_weights, weights], 1)
    weights_run = sess.run(concatenated_weights)
    record_shape = weights_run.shape[1]

    f = tables.open_file(weights_history_dump_path, mode='w')
    atom = tables.Float64Atom()
    file_array = f.create_earray(f.root, 'data', atom, (0, record_shape))
    file_array.append(weights_run)

    print('Starting Training...')
    total_loss = 0
    for i in range(total_iteration):
        weights_list, loss, _ = sess.run([concatenated_weights, n0.ops_loss_train, n0.ops_min_step], feed_dict={n0.drop_out_prob: 0.5})
        file_array.append(weights_list)
        total_loss += loss
        if (i + 1) % print_interval == 0:
            print('Iteration ' + str(i + 1) + ' / ' + str(total_iteration))
            avg_log_loss = np.log10(total_loss / print_interval)
            print('Log Loss: ', avg_log_loss)
            with open(loss_file_path, 'a') as loss_file_handle:
                loss_file_handle.write(str(avg_log_loss) + ' ')
                loss_file_handle.write('\n')






