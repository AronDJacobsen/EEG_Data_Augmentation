

import numpy as np
import torch
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pickle as pkl
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def subset(X, y, ID_frame, no_indiv):

    individuals = np.unique(ID_frame)

    new_indiv = set(np.random.choice(individuals, no_indiv, replace=False))
    #Splitting train and test set.
    indices = [i for i, ID in enumerate(ID_frame) if ID in new_indiv]

    new_X, new_y, new_ID_frame = X[indices,:], y[indices], ID_frame[indices]

    return new_X, new_y, new_ID_frame


def binary(X, y, ID_frame):

    classes = len(y[0,:])

    # where only one of the classes are present (except null)
    transform_indices = np.where(np.sum(y[:,:classes-1],axis=1) == 1)[0]

    # we one is present, we set is to 0 in the null class
    y[transform_indices, classes-1] = np.zeros(len(transform_indices))

    # we now only include where one class is present
    include = np.where(np.sum(y[:,:classes],axis=1) == 1)[0]
    y = y[include, :]
    X = X[include, :]
    ID_frame = ID_frame[include]

    '''
    #indices with more than 1 class
    del_indices = np.where(np.sum(y[:,:classes],axis=1) > 1)[0]

    X = np.delete(X, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    ID_frame = np.delete(ID_frame,del_indices, axis=0)
    '''

    return X, y, ID_frame



def smote(X, y, multi):

    #undersample majority
    #balance_b = Counter(y)
    if multi:
        lb = preprocessing.LabelBinarizer()
        y = np.argmax(y, axis=1)


    #oversample minority
    over = SMOTE() # increase minority to have % of majority

    X_over, y_over = over.fit_resample(X, y)
    # how is the balance now?
    #balance_a = Counter(y)

    #print('Before:', balance_b)
    #print('After: ', balance_a)

    if multi:
        y_over = lb.fit_transform(y_over)



    return X_over, y_over



def rand_undersample(X, y, arg, state, multi):

    if multi:
        lb = preprocessing.LabelBinarizer()
        y = np.argmax(y, axis=1)
        under = RandomUnderSampler(sampling_strategy=arg, random_state = state)
        X_under, y_under = under.fit_resample(X, y)
        y_under = lb.fit_transform(y_under)
    else:
        #undersample majority
        #balance_b = Counter(y) # for binary
        # https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
        under = RandomUnderSampler(sampling_strategy=arg, random_state = state)
        X_under, y_under = under.fit_resample(X, y)
        # how is the balance now?

        #balance_a = Counter(y)
        #print('Before:', balance_b)
        #print('After: ', balance_a)


    return X_under, y_under


def nearmiss(X, y, version, n_neighbors):

    #undersample majority
    #balance_b = Counter(y)
    under = RandomUnderSampler(sampling_strategy=reduce) # reduce majority to have % more than minority
    X_under, y_under = under.fit_resample(X, y)
    # how is the balance now?
    #balance_a = Counter(y)

    #print('Before:', balance_b)
    #print('After: ', balance_a)


    return X_under, y_under





def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# loss based on mixup
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#%%

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
