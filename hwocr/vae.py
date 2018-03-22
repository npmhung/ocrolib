'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''

from scipy.stats import norm
import os
import sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../')
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import metrics
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from ocrolib.hwocr import run_hw2
from keras.models import model_from_json
from keras.datasets import mnist
import numpy as np
import cv2
import random
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


img_rows, img_cols, img_chns = 64, 64, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3
channel_first = K.image_data_format() == 'channels_first'
# channel_first = True

if channel_first:
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

batch_size = 512
original_dim = img_rows*img_cols
latent_dim = 16
intermediate_dim = 200
epochs = 50
epsilon_std = 1.0

def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print('x_train.shape:', x_train.shape)
    return x_train, x_test, y_train, y_test

def get_mnist_cnn():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    print('x_train.shape:', x_train.shape)
    return x_train, x_test, y_train, y_test

def get_etl_cnn(data_dir='./data/full_katakana_quote.nice.pkl'):
    x_train, y_train, x_test, y_test = pickle.load(open(data_dir, 'rb'))
    print('x_train.shape:', x_train.shape)
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
    return x_train, x_test, y_train, y_test

def build_model():
    x = Input(shape=(original_dim,))
    encoder_h = Dense(intermediate_dim, activation='relu')(x)
    # 2 outputs
    z_mean = Dense(latent_dim)(encoder_h)
    z_log_var = Dense(latent_dim)(encoder_h)

    # sampleing z from mean and variance computed by neural net
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon  # reparameter trick

    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            # these two loss exists to miminize KL loss between the approximate oster q(z|x)-the encoder and the real p(z|x)
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)  # reconstruct loss
            # this KL is between the q(z|x) and p(z) both assumed to be unit gaussian
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  # regularizer loss
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)  # api to make loss
            # We won't actually use the output.
            return x

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)

    encoder = Model(x, z_mean)

    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)#borrow the learn layer of vae

    return vae,encoder, generator


def build_model_cnn():
    x = Input(shape=original_img_size)
    conv_1 = Conv2D(img_chns,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * img_cols//2 * img_rows//2, activation='relu')

    if channel_first:
        output_shape = (batch_size, filters, img_cols//2, img_rows//2)
    else:
        output_shape = (batch_size, img_cols//2, img_cols//2, filters)

    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    if channel_first:
        output_shape = (batch_size, filters, img_cols+1, img_rows+1)
    else:
        output_shape = (batch_size, img_cols+1, img_rows+1, filters)

    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash = Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')

    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean_squash):
            x = K.flatten(x)
            x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
            xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean_squash = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean_squash)
            self.add_loss(loss, inputs=inputs)
            # We don't use this output.
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean_squash])
    vae = Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None)
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
    generator = Model(decoder_input, _x_decoded_mean_squash)

    return vae, encoder, generator

def build_denoise_model_cnn():
    x = Input(shape=original_img_size)
    x2 = Input(shape=original_img_size)
    conv_1 = Conv2D(img_chns,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * img_cols//2 * img_rows//2, activation='relu')

    if channel_first:
        output_shape = (batch_size, filters, img_cols//2, img_rows//2)
    else:
        output_shape = (batch_size, img_cols//2, img_cols//2, filters)

    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    if channel_first:
        output_shape = (batch_size, filters, img_cols+1, img_rows+1)
    else:
        output_shape = (batch_size, img_cols+1, img_rows+1, filters)

    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash = Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')

    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean_squash):
            x = K.flatten(x)
            x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
            xent_loss = img_cols * img_rows  * metrics.binary_crossentropy(x, x_decoded_mean_squash)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean_squash = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean_squash)
            self.add_loss(loss, inputs=inputs)
            # We don't use this output.
            return x

    y = CustomVariationalLayer()([x2, x_decoded_mean_squash])
    vae = Model([x,x2], y)
    vae.compile(optimizer='rmsprop', loss=None)
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
    generator = Model(decoder_input, _x_decoded_mean_squash)

    return vae, encoder, generator

def train_nvae(data_dir='./data/full_katakana_quote.nice.pkl',
               ndata_dir='./data/full_katakana_quote.nice.noise.pkl',
               save_m='./save/vae/model_kata.noise.json',
              save_w='./save/vae/weight_kata.noise.h5'):
    vae, _,_ =build_denoise_model_cnn()
    vae.compile(optimizer='rmsprop', loss=None)

    print(vae.summary())
    model_json = vae.to_json()
    with open(save_m, "w") as json_file:
        json_file.write(model_json)

    checkpoint = ModelCheckpoint(save_w, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    # train the VAE on MNIST digits
    x_train, x_test, y_train, y_test =get_etl_cnn(data_dir)
    nx_train, nx_test, ny_train, ny_test = get_etl_cnn(ndata_dir)
    vae.fit([nx_train,x_train],
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint],
            validation_data=([nx_test,x_test], None))

def test_vae(data_dir='./data/full_katakana_quote.nice.noise.pkl',save_w='./save/vae/weight_kata.noise.h5'):
    nx_train, nx_test, ny_train, ny_test = get_etl_cnn(data_dir)
    vae, encoder, generator=build_denoise_model_cnn()

    run_hw2.load_model_weights(save_w, vae)
    print(vae.summary())


    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(nx_test, batch_size=batch_size)
    fig=plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=ny_test)
    plt.colorbar()
    # plt.show()
    fig.savefig('test_images/nvae/{}.report.jpg'.format('latent_space'))

    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    fig=plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    # plt.show()
    fig.savefig('test_images/nvae/{}.report.jpg'.format('gen_im'))


def test_nvae(data_dir='./data/full_katakana_quote.nice.noise.pkl',
              save_w='./save/vae/weight_kata.noise.h5',
              store_dir='./test_images/nvae/'):
    nx_train, nx_test, ny_train, ny_test = get_etl_cnn(data_dir)
    vae, encoder, generator=build_denoise_model_cnn()

    run_hw2.load_model_weights(save_w, vae)
    print(vae.summary())


    # display a 2D plot of the digit classes in the latent space
    x_test_denoise = generator.predict(encoder.predict(nx_test, batch_size=batch_size, verbose=1),verbose=1)\
        .reshape(nx_test.shape[0],img_rows,img_cols)
    indexs = list(range(nx_test.shape[0]))
    random.shuffle(indexs)
    num=100
    nx_test=nx_test.reshape(nx_test.shape[0],img_rows,img_cols)
    # dir1=store_dir+'/noise/'
    # if not os.path.isdir(dir1):
    #     os.mkdir(dir1)

    # for ci,i in enumerate(indexs):
    #     im=nx_test[i]*255
    #     # print(np.max(im))
    #     # print(np.min(im))
    #     cv2.imwrite('{}/{}.png'.format(dir1,ci),im)
    #     if ci>num:
    #         break
    dir1 = store_dir + '/denoise/'
    if not os.path.isdir(dir1):
        os.mkdir(dir1)
    dir1+=save_w.split(os.sep)[-1][6:-2]
    if not os.path.isdir(dir1):
        os.mkdir(dir1)
    for ci, i in enumerate(indexs):
        im = x_test_denoise[i]*255
        im2 = nx_test[i] * 255
        ret, im = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        a.set_title('Noise')
        plt.imshow(im2, cmap='gray')
        a = fig.add_subplot(1, 2, 2)
        a.set_title('Denoise')
        plt.imshow(im, cmap='gray')
        fig.savefig('{}/{}.png'.format(dir1, ci))
        plt.close()
        if ci > num:
            break


if __name__=='__main__':
    # train_nvae(data_dir='./data/full_katakana_quote.nice.pkl',ndata_dir='./data/full_katakana_quote.nice.noise.pkl',
    #            save_m='./save/vae/model_kata.noise.json',save_w='./save/vae/weight_kata.noise.h5')

    # train_nvae(data_dir='./data/hiragana.nice.pkl', ndata_dir='./data/hiragana.nice.noise.pkl',
    #            save_m='./save/vae/model_hiragana.noise.json', save_w='./save/vae/weight_hiragana.noise.h5')

    # test_nvae(data_dir='./data/full_katakana_quote.nice.noise.pkl',
    #           save_w='./save/vae/weight_kata.noise.h5')

    # test_nvae(data_dir='./data/hiragana.nice.noise.pkl',
    #           save_w='./save/vae/weight_hiragana.noise.h5')

    # train_nvae(data_dir='./data/kanji.nice.pkl', ndata_dir='./data/kanji.nice.noise.pkl',
    #            save_m='./save/vae/model_kanji.noise.json', save_w='./save/vae/weight_kanji.noise.h5')

    test_nvae(data_dir='./data/kanji.nice.noise.pkl',
              save_w='./save/vae/weight_kanji.noise.h5')
