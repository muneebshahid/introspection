import tables
import numpy as np
import tensorflow as tf
from networks import IntrospectionNetwork


def write_to_file(f_name, list_var):
    with open(f_name, 'a') as log_file:
        for variable in list_var:
            log_file.write(str(variable) + ' ')
        log_file.write('\n')

train_data_path = 'log_folder/mnist_train_data.h5'
train_data_file = tables.open_file(train_data_path, 'r')
train_data = train_data_file.root.data

training_itr = 30000
print_interval = 2
scaling_coefficient = 1000
save_interval = 6

ins = IntrospectionNetwork()
ins.build()

# since we predict 2t, make sure all ts have corresponding tragets.
maxval = train_data.shape[0] / 2
data_width = train_data.shape[1]
data_0 = train_data[0, :]

def get_batch():
    batch_indices = np.zeros(20, dtype=np.int32)
    t = np.random.randint(low=1, high=maxval)
    t_1 = int(7 * t / 10)
    t_2 = int(4 * t / 10)
    data_t = train_data[t, :]
    data_t_1 = train_data[t_1, :]
    data_t_2 = train_data[t_2, :]
    diff = np.abs(data_0 - data_t)
    sorted_diff = np.argsort(diff)
    p_50_100 = sorted_diff[data_width / 2: ]
    p_25_50 = sorted_diff[data_width / 4: data_width / 2]
    p_0_25 = sorted_diff[: data_width / 4]

    batch_indices[:10] = np.random.choice(p_50_100, 10, replace=False)
    batch_indices[10:15] = np.random.choice(p_25_50, 5, replace=False)
    batch_indices[15:] = np.random.choice(p_0_25, 5, replace=False)
    batch_t = np.take(data_t, batch_indices)

    batch_t_1 = np.take(data_t_1, batch_indices)
    batch_t_2 = np.take(data_t_2, batch_indices)
    batch_t_0 = np.take(data_0, batch_indices)
    labels = np.take(train_data[2 * t, :], batch_indices)[np.newaxis, :]

    batch = np.vstack((batch_t, batch_t_1))
    batch = np.vstack((batch, batch_t_2))
    batch = np.vstack((batch, batch_t_0))

    return batch, labels


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_loss = 0
    print('Starting Trainig')
    for i in range(training_itr):
        inputs, labels = get_batch()
        _, loss, _ = sess.run([ins.ops_optim_step, ins.ops_loss, ins.ops_inc_global_step], feed_dict={ins.inputs: inputs * scaling_coefficient,
                                                                          ins.targets: labels * scaling_coefficient})
        total_loss += loss
        if (i + 1) % print_interval == 0:
            avg_loss = total_loss / print_interval
            print('Iteration: ' + str(i + 1) + ' / ' + str(training_itr))
            print('loss: ' + str(avg_loss))
            write_to_file('log_folder/introspection_loss' , [loss])
            total_loss = 0

        if (i + 1) % save_interval == 0:
            print('Saving Network')
            ins.save(sess, 'introspection_networks/ins')


