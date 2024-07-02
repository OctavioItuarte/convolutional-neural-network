import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():

    model = models.Sequential()

    model.add(layers.RandomFlip("horizontal_and_vertical", input_shape=(32, 32, 3)))
    model.add(layers.RandomRotation(0.2))
    model.add(layers.RandomContrast(factor=0.2))
    model.add(layers.RandomCrop(height=28, width=28))
    model.add(layers.RandomZoom(height_factor=0.2, width_factor=0.2))

    model.add(layers.Conv2D(64, kernel_size=(5, 5), padding= 'same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, kernel_size=(5, 5), padding= 'same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), padding= 'same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), padding= 'same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    return model