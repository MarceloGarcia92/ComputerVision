from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import os

# Defining env variables
PATH = 'Data/plates/plates/'
TRAIN_PATH = f'{PATH}/train'
TEST_PATH = f'{PATH}/test'
VAL_PATH = f'{PATH}/valid'
TARGET_SIZE = (150, 150, 3)

# Creation of variables
n_labels = len(os.listdir(TRAIN_PATH))
print(f'There are {n_labels} labels')

directories = [dir for dir in os.listdir(TRAIN_PATH) if os.path.isdir(f'{TRAIN_PATH}/{dir}')]
print(f'There directories in the data are: {directories}')


def gen_creator(target_size, train_path, test_path, val_path):
    # Creation of the Generators
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Creation of the datasets Generators
    train_set = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size[:2],
        batch_size=20,
        class_mode='categorical',
        seed=42)

    val_set = train_datagen.flow_from_directory(
        val_path,
        target_size=target_size[:2],
        batch_size=20,
        class_mode='categorical',
        seed=42)

    test_set = test_datagen.flow_from_directory(
        test_path,
        target_size=target_size[:2],
        batch_size=20,
        class_mode='categorical',
        seed=42)

    return train_set, val_set, test_set


# model creation
def transfer_learning(input_shape, n_labels, vgg16=None):
    if n_labels > 1:
        output_act = 'softmax'
    else:
        output_act = 'sigmoid'
    if vgg16:
        from tensorflow.keras.applications.vgg16 import VGG16

        model_vgg = VGG16(include_top=False, weights='imagenet', input_tensor=None,
                      input_shape=input_shape, pooling=None)

        for layer in model_vgg.layers:
            layer.trainable = False

        x = Flatten()(model_vgg.output)
        x = Dense(n_labels, activation=output_act)(x)

    model = Model(inputs=model_vgg.input, outputs=x)

    return model


train_set, val_set, test_set = gen_creator(TARGET_SIZE, TRAIN_PATH, TEST_PATH, VAL_PATH)

model = transfer_learning(TARGET_SIZE, n_labels, vgg16=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(train_set, validation_data=val_set, batch_size=20, epochs=5)
