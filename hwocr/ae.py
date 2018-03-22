
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from ocrolib.hwocr import run_hw2
from keras.models import model_from_json
from keras.datasets import mnist
import numpy as np
import cv2
import random

def train_ae(save_m='./save/ae/model_mnist.json',save_w='./save/ae/weight_mnist.h5'):
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # this model maps an input to its encoded representation

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




    (x_train, _), (x_test, _) = mnist.load_data()



    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    print (x_train.shape)
    print (x_test.shape)


    print(autoencoder.summary())
    model_json = autoencoder.to_json()
    with open(save_m, "w") as json_file:
        json_file.write(model_json)

    checkpoint = ModelCheckpoint(save_w, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=1024,callbacks= [checkpoint],
                    shuffle=True,
                    validation_data=(x_test, x_test))

def test_ae(save_m='./save/ae/model_mnist.json', save_w='./save/ae/weight_mnist.h5'):
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    run_hw2.load_model_weights(save_w, model)
    print(model.summary())


    import matplotlib.pyplot as plt
    decoded_imgs = model.predict(x_test)

    nl = list(range(x_test.shape[0]))
    n=10
    random.shuffle(nl)
    plt.figure(figsize=(20, 4))
    for i, id in enumerate(nl[:n]):
        # display original
        ax = plt.subplot(2, n, i+1)
        im=x_test[id].reshape(28, 28)*255
        _, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY_INV)
        plt.imshow(im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        im=decoded_imgs[id].reshape(28, 28)*255
        _,im = cv2.threshold(im,100,255,cv2.THRESH_BINARY_INV)
        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        plt.imshow(im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__=='__main__':
    # train_ae()
    test_ae()
