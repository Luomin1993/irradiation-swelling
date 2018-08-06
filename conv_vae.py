from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Reshape
from keras.models import Model,load_model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import img_data


def self_load_data(path='mnist.npz'):
    #Loads the MNIST dataset.
    path='./mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

#(x_train, y_train), (x_test, y_test) = self_load_data()
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


#x_train_out = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test_out = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#x_train_conv = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test_conv = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format



x_train,y_train = img_data.make_Xy()

x_train = x_train.astype('float32')*255 #/ 255.
#x_train = np.reshape(x_train, (len(x_train), 28,28, 1))  # adapt this if using `channels_first` image data format
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
y_train = np.reshape(y_train, (len(y_train), 32))

#-------------------------en de------------------------
# input_img = Input(shape=(28,28, 1))  # adapt this if using `channels_first` image data format

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x) # at this point the representation is (4, 4, 8) i.e. 128-dimensional
# x = Flatten()(x)
# #x = Reshape((128,-1))(x)
# x = Dense(64, activation='relu')(x) 
# encoded = Dense(32, activation='relu')(x)

# input_con = Input(shape=(32,))
# #decoded = Dense(784, activation='sigmoid')(decoded)
# #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# x = Dense(64, activation='relu')(input_con)
# x = Dense(128, activation='relu')(x)
# x = Reshape((4, 4, 8))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

#-------------------------auto en de cnn------------------------
# input_img = Input(shape=(28,28, 1))  # adapt this if using `channels_first` image data format
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

#-------------------------auto en de dnn------------------------
input_img = Input(shape=(784,))
encoded = Dense(392, activation='relu')(input_img)
encoded = Dense(196, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(196, activation='relu')(decoded)
decoded = Dense(392, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)


# encoder = Model(input_img, encoded)
# encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# from keras.callbacks import TensorBoard

# encoder.fit(x_train, y_train,
#                 epochs=5,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_train[0:100], y_train[0:100]))
# encoder.save('./models/model_ir.h5')


# decoder = Model(input_con, decoded)
# decoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# decoder.fit(y_train, x_train,
#                 epochs=5,
#                 batch_size=50,
#                 shuffle=True,
#                 validation_data=(y_train[0:100], x_train[0:100]))
# decoder.save('./models/model_de.h5')



# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.fit(x_train, x_train,
#                 epochs=5,
#                 batch_size=250,
#                 shuffle=True,
#                 validation_data=(x_train[0:100], x_train[0:100]))
# autoencoder.save('./models/auto.h5')


#------------------test------------
x_test = x_train[7000:7013]
y_test = y_train[7000:7013]
autoencoder  = load_model('./models/auto.h5');
decoded_imgs = autoencoder.predict(x_test)

print(x_test)
print('----------------------------------')
print(decoded_imgs)


import matplotlib.pyplot as plt


n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow((255*(decoded_imgs[i]>decoded_imgs.mean())).reshape(28, 28))
    #plt.imshow((decoded_imgs[i]*3).reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()                
