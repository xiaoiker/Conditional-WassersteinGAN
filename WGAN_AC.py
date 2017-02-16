from __future__ import print_function

import cv2
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.examples.tutorials.mnist import input_data


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generator_conv(z):
    train = ly.fully_connected(z,
                               4 * 4 * 512, activation_fn=tf.nn.relu, normalizer_fn=None)
    train = tf.reshape(train, (-1, 512, 4, 4))
    train = ly.conv2d_transpose(train, 256, 3, stride=2, data_format=data_format,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME',
                                normalizer_params={'fused': True, 'data_format': data_format},
                                weights_initializer=tf.random_normal_initializer(0, 0.04))
    train = ly.conv2d_transpose(train, 128, 3, stride=2, data_format=data_format,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME',
                                normalizer_params={'fused': True, 'data_format': data_format},
                                weights_initializer=tf.random_normal_initializer(0, 0.04))
    train = ly.conv2d_transpose(train, 64, 3, stride=2, data_format=data_format,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME',
                                normalizer_params={'fused': True, 'data_format': data_format},
                                weights_initializer=tf.random_normal_initializer(0, 0.04))
    train = ly.conv2d_transpose(train, channel, 3, stride=1, data_format=data_format,
                                activation_fn=tf.nn.tanh, padding='SAME',
                                weights_initializer=tf.random_normal_initializer(0, 0.04))
    return train


def critic_conv(x, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64

        img = ly.conv2d(x, num_outputs=size, kernel_size=3, data_format=data_format,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3, data_format=data_format,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                        normalizer_params={'fused': True, 'data_format': data_format})
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3, data_format=data_format,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                        normalizer_params={'fused': True, 'data_format': data_format})

        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3, data_format=data_format,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                        normalizer_params={'fused': True, 'data_format': data_format})
        img = tf.reshape(img, [batch_size, -1])

        disc = ly.fully_connected(img, 1, activation_fn=None)
    return disc


def classifier_conv(x, reuse=False):
    with tf.variable_scope('classifier') as scope:
        if reuse:
            scope.reuse_variables()
        size = 32

        img = ly.conv2d(x, num_outputs=size, kernel_size=5, data_format=data_format,
                        stride=1, activation_fn=tf.nn.relu)
        img = ly.max_pool2d(img, 2, stride=2, padding='SAME', data_format=data_format)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=5, data_format=data_format,
                        stride=1, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm,
                        normalizer_params={'fused': True, 'data_format': data_format})
        img = ly.max_pool2d(img, 2, stride=2, padding='SAME', data_format=data_format)

        img = tf.reshape(img, [batch_size, -1])

        # fully connect
        img = ly.fully_connected(img, 1024, activation_fn=tf.nn.relu)

        cat = ly.fully_connected(img, y_dim, activation_fn=None)
    return cat


def build_graph(is_test=False):

    ############################ Inputs ##################################
    # real data
    real_data = tf.placeholder(
        dtype=tf.float32, shape=(batch_size, channel, img_dim, img_dim))
    real_label = tf.placeholder(dtype=tf.float32, shape=(batch_size, y_dim))

    # fake data
    z_cat = tf.placeholder(dtype=tf.float32, shape=(batch_size, y_dim))
    z_rand = tf.placeholder(tf.float32, shape=(batch_size, z_dim))
    z = tf.concat([z_cat, z_rand], axis=1)
    ########################### End Inputs ###############################

    ############################# Graph #################################
    generator = generator_conv
    discriminator = critic_conv
    classifier = classifier_conv

    # Generator
    with tf.variable_scope('generator'):
        gen = generator(z)
    if is_test:
        return gen, z_cat, z_rand

    # image summary
    img_sum = tf.summary.image("img", tf.transpose(gen, (0, 2, 3, 1)), max_outputs=10)

    # Discriminator
    disc_real = discriminator(real_data)
    disc_fake = discriminator(gen, reuse=True)

    # Classifier
    cat_real = classifier(real_data)
    cat_fake = classifier(gen, reuse=True)

    # Loss
    # Wasserstein
    d_loss = tf.reduce_mean(disc_fake - disc_real)
    g_loss = tf.reduce_mean(-disc_fake)
    g_loss_sum = tf.summary.scalar("wasserstein_loss_g", g_loss)
    c_loss_sum = tf.summary.scalar("wasserstein_loss_d", d_loss)

    # Categorical Loss
    loss_c_f = tf.nn.softmax_cross_entropy_with_logits(logits=cat_fake, labels=z_cat)
    loss_c_f_sum = tf.summary.scalar("categorical_loss_c_fake", tf.reduce_mean(loss_c_f))
    loss_c_r = tf.nn.softmax_cross_entropy_with_logits(logits=cat_real, labels=real_label)
    loss_c_r_sum = tf.summary.scalar("categorical_loss_c_real", tf.reduce_mean(loss_c_r))
    loss_c = (loss_c_r + loss_c_f) / 2
    loss_c_sum = tf.summary.scalar("categorical_loss_c", tf.reduce_mean(loss_c))
    ############################# End Graph ##############################

    ############################# Optimization #################################
    # Variable Collections
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    theta_c = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

    # Optimizers
    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = optimize(loss=g_loss, learning_rate=learning_rate_ger,
                     optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer,
                     variables=theta_g, global_step=counter_g,
                     summaries='gradient_norm')
    counter_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_d = optimize(loss=d_loss, learning_rate=learning_rate_dis,
                     optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer,
                     variables=theta_d, global_step=counter_d,
                     summaries='gradient_norm')
    counter_c_f = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_f = optimize(loss=loss_c_f, learning_rate=learning_rate_ger,
                       optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer,
                       variables=theta_g, global_step=counter_c_f,
                       summaries='gradient_norm')
    counter_c_r = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_r = optimize(loss=loss_c, learning_rate=learning_rate_cat,
                       optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer,
                       variables=theta_c, global_step=counter_c_r,
                       summaries='gradient_norm')

    # Clip weights
    clipped_var_d = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_d]
    # merge the clip operations on discriminator variables
    with tf.control_dependencies([opt_d]):
        opt_d = tf.tuple(clipped_var_d)
    ########################### End Optimization ################################

    return opt_g, opt_d, opt_c_f, opt_c_r, real_data, real_label, z_cat, z_rand


def optimize(loss, learning_rate, optimizer, variables, global_step, summaries):
    """Modified from sugartensor"""

    optim = optimizer(learning_rate=learning_rate)

    # Calculate Gradient
    gradients = optim.compute_gradients(loss, var_list=variables)

    # Add Summary
    if summaries is None:
        summaries = ["loss", "learning_rate"]
    if "gradient_norm" in summaries:
        tf.summary.scalar("global_norm/gradient_norm",
                          clip_ops.global_norm(list(zip(*gradients))[0]))
    # Add scalar summary for loss.
    if "loss" in summaries:
        tf.summary.scalar("loss", loss)

    # Add histograms for variables, gradients and gradient norms.
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        if grad_values is not None:
            var_name = variable.name.replace(":", "_")
            if "gradients" in summaries:
                tf.summary.histogram("gradients/%s" % var_name, grad_values)
            if "gradient_norm" in summaries:
                tf.summary.scalar("gradient_norm/%s" % var_name,
                                  clip_ops.global_norm([grad_values]))

    # Gradient Update OP
    return optim.apply_gradients(gradients, global_step=global_step)


# Global parameters
batch_size = 64
img_dim = 32
z_dim = 128
y_dim = 10
learning_rate_ger = 2e-4
learning_rate_dis = 2e-4
learning_rate_cat = 2e-4
device = '/gpu:0'
data_format = 'NCHW'
# img size
s = 32
# update Diters times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)
c_interv = 2.0
Diters = 5
Diters = int((c_interv + 1) / c_interv * Diters)
# the upper bound and lower bound of parameters in critic
clamp_lower = -0.01
clamp_upper = 0.01
# whether to use adam for parameter update, if the flag is set False, use tf.train.RMSPropOptimizer
# as recommended in paper
is_adam = False
channel = 1
# directory to store log, including loss and grad_norm of generator and critic
log_dir = './log_wgan'
ckpt_dir = './ckpt_wgan'
classifier_dir = './cat_wgan'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(classifier_dir):
    os.makedirs(classifier_dir)
# max iter step, note the one step indicates that a Citers updates of critic and one update of generator
max_iter_step = 20000


def main():
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.device(device):
        opt_g, opt_d, opt_c_f, opt_c_r, real_data, real_label, z_cat, z_rand = build_graph()

    merged_all = tf.summary.merge_all()

    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    def next_feed_dict(iter):
        train_img, train_label = dataset.train.next_batch(batch_size)
        train_img = 2 * train_img - 1

        train_img = np.reshape(train_img, (-1, channel, 28, 28))
        npad = ((0, 0), (0, 0), (2, 2), (2, 2))
        train_img = np.pad(train_img, pad_width=npad,
                           mode='constant', constant_values=-1)
        batch_z_rand = np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)

        # Generate random one-hot vectors as class condition
        if iter % 2 == 0:
            idx = np.random.random_integers(0, y_dim - 1, size=(batch_size,))
        else:
            idx = np.random.random_integers(0, y_dim - 1)
        y_generated = np.zeros((batch_size, y_dim))
        y_generated[np.arange(batch_size), idx] = 1

        feed_dict = {real_data: train_img, real_label: train_label, z_cat: y_generated, z_rand: batch_z_rand}
        return feed_dict

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        for i in range(max_iter_step):
            if i % 100 == 0:
                print(i)

            if i < 25 or i % 500 == 0:
                diters = 100
            else:
                diters = Diters * 2

            # Train Discriminator
            for j in range(diters):
                feed_dict = next_feed_dict(i)
                if i % 100 == 99 and j == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.NO_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, merged = sess.run([opt_d, merged_all], feed_dict=feed_dict,
                                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'discriminator_metadata {}'.format(i), i)
                else:
                    sess.run(opt_d, feed_dict=feed_dict)

            # Train Generator
            feed_dict = next_feed_dict(i)
            if i % 100 == 99:
                _, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'generator_metadata {}'.format(i), i)
            else:
                sess.run(opt_g, feed_dict=feed_dict)

            # Train Generator on classifier
            if i % c_interv == 0:
                feed_dict = next_feed_dict(i)
                if i % 100 == 99:
                    _, merged = sess.run([opt_c_f, merged_all], feed_dict=feed_dict,
                                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'classifier_f_metadata {}'.format(i), i)
                else:
                    sess.run(opt_c_f, feed_dict=feed_dict)

            # Train Classifier on fake and real
            if i % c_interv == 0:
                feed_dict = next_feed_dict(i)
                if i % 100 == 99:
                    _, merged = sess.run([opt_c_r, merged_all], feed_dict=feed_dict,
                                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'classifier_r_metadata {}'.format(i), i)
                else:
                    sess.run(opt_c_r, feed_dict=feed_dict)

            # Save model
            if i % 1000 == 999:
                saver.save(sess, os.path.join(
                    ckpt_dir, "model.ckpt"), global_step=i)


def test(test_num=10):
    img = np.zeros((img_dim * test_num, img_dim * test_num, channel))

    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.device(device):
            train, z_cat, z_rand = build_graph(is_test=True)
        saver = tf.train.Saver()

        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        for i in xrange(10):
            batch_z = np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
            # Generate one-hot vectors as class condition
            y_generated = np.zeros((batch_size, y_dim))
            y_generated[:, i] = 1

            output = sess.run(tf.transpose(train, (0, 2, 3, 1)), feed_dict={z_rand: batch_z, z_cat: y_generated})

            img[:, img_dim*i:img_dim*(i+1), :] = np.reshape(output[:test_num, :, :, :], (test_num * img_dim, img_dim, channel))

    image = ((img / 2 + 0.5)*255).astype(np.uint8)
    cv2.imshow("test", image)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
    # test()
