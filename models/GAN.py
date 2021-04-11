import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# TODO: HVORFOR BRUGER VI PKL? David siger at npy er langt hurtigere
# GAN
# https://github.com/mchablani/deep-learning/blob/master/gan_mnist/Intro_to_GANs_Solution.ipynb

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name="inputs_real")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="inputs_z")

    return inputs_real, inputs_z


def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    ''' Build the generator network.

        Arguments
        ---------
        z : Input tensor for the generator
        out_dim : Shape of the generator output
        n_units : Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU

        Returns
        -------
        out, logits:
    '''
    with tf.variable_scope('generator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(z, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(h1, alpha*h1)

        # Logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.nn.tanh(logits)

        return out, logits


def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    ''' Build the discriminator network.

        Arguments
        ---------
        x : Input tensor for the discriminator
        n_units: Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU

        Returns
        -------
        out, logits:
    '''
    with tf.variable_scope('discriminator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(h1, alpha*h1)

        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.nn.sigmoid(logits)

        return out, logits


def GAN(X, NtoGenerate, z_size = 100, g_hidden_size = 128, d_hidden_size = 128, alpha = 0.01, smooth = 0.1, learning_rate = 0.0002): # Should be used on each of the binary classes.
    # X[np.where(y[:, 1] == 1)]

    # Hyperparameters
    # Size of input image to discriminator
    input_size = 475  # size of each window
    # Size of latent vector to generator, typically 100, however NVIDIA used N equal to size of max number of channels in the convolutions
    # z_size = 100
    # Sizes of hidden layers in generator and discriminator
    #g_hidden_size = 128
    #d_hidden_size = 128
    # Leak factor for leaky ReLU
    #alpha = 0.01
    # Label smoothing
    #smooth = 0.1

    tf.reset_default_graph()
    # Create our input placeholders
    input_real, input_z = model_inputs(input_size, z_size)

    # Generator network here
    g_model, g_logits = generator(input_z, input_size, n_units = g_hidden_size, reuse=False, alpha=alpha)
    # g_model is the generator output

    # Disriminator network here
    d_model_real, d_logits_real = discriminator(input_real, n_units = d_hidden_size, reuse=False, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model, n_units = d_hidden_size, reuse=True, alpha=alpha)

    # Calculate losses
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_logits_real) * (1 - smooth)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.zeros_like(d_logits_real)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_logits_fake)))


    # Get the trainable_variables, split into G and D parts
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    # TRAINING
    batch_size = 100 # TODO: Might have to be different size according to the size of the class (This determines amount of generated data)
    epochs = 100
    samples = []
    losses = []
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(len(X) // batch_size):
                #print(batch_size * ii, batch_size * (ii + 1))
                batch = X[batch_size * ii : batch_size * (ii + 1)]

                # Get images, reshape and rescale to pass to D
                # batch_images = batch[0].reshape((batch_size, 475))

                # The images should be rescaled to be between -1 and 1, as tanh works best. (Rescale back afterwards?)
                batch_images = (batch - np.min(batch)) / (np.max(batch) - np.min(batch)) * (1 - (-1)) + -1
                # When rescaling back:
                # -(np.max(batch)*(-1) - (np.max(batch)*batch_images) - (np.min(batch)*1) + np.min(batch)*batch_images)/(1 - (-1))

                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

                # Run optimizers
                _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})

            # At the end of each epoch, get the losses and print them out
            train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
            train_loss_g = g_loss.eval({input_z: batch_z})

            print("Epoch {}/{}...".format(e + 1, epochs),
                  "Discriminator Loss: {:.4f}...".format(train_loss_d),
                  "Generator Loss: {:.4f}".format(train_loss_g))
            # Save losses to view after training
            losses.append((train_loss_d, train_loss_g))

            # Might be unnecessary
            # Sample from generator as we're training for viewing afterwards
            sample_z = np.random.uniform(-1, 1, size=(16, z_size))
            gen_samples = sess.run(
                generator(input_z, input_size, n_units = g_hidden_size, reuse=True),
                feed_dict={input_z: sample_z})
            samples.append(gen_samples)
            saver.save(sess, './checkpoints/generator.ckpt')

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()

    # Generating the new observations after training:
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        sample_z = np.random.uniform(-1, 1, size=(NtoGenerate, z_size))
        gen_samples = sess.run(
            generator(input_z, input_size, n_units = g_hidden_size, reuse=True),
            feed_dict={input_z: sample_z})

    # Scaling back to normal:
    gen_samples = 1/2*gen_samples[0]*np.max(X) - 1/2*gen_samples[0]*np.min(X) + 1/2*np.max(X) + 1/2*np.min(X)

    return gen_samples

# example:
# gen_samples = GAN(X[np.where(y[:,2] == 1)], 10000)

