import tensorflow.keras as K
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = np.asarray([[1 if i==int(j) else 0 for i in range(10)] for j in y_train]).astype('float32')
y_test = np.asarray([[1 if i==int(j) else 0 for i in range(10)] for j in y_test]).astype('float32')

downscaled_model = K.models.Sequential()
downscaled_model.add(K.layers.AveragePooling2D((4,4), input_shape=(28,28,1)))

downscale = downscaled_model.predict(np.reshape(x_train, (60000,28,28,1)))

decoder = K.models.Sequential()
decoder.add(K.layers.Flatten(input_shape=(7,7,1)))
decoder.add(K.layers.Dense(128, activation='relu'))
decoder.add(K.layers.Dense(512, activation='relu'))
decoder.add(K.layers.Dense(784, activation='sigmoid'))

decoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
decoder.fit(downscale, x_train, epochs=30, batch_size=128, shuffle=True)


test_input = np.asarray([[1 if i == j else 0 for i in range(10)] for j in range(10)]).astype('float32')
decoded_imgs = decoder.predict(downscale)
print(test_input)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(downscale[i].reshape(7, 7))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()