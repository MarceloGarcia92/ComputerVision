from keras.applications import VGG16
from keras import models
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential


def freeze(conv_base, layer='block5_conv1'):
    '''Freezing all layers up to a specific one'''
    conv_base.trainable = True

    set_trainable = False

    for layer in conv_base.layers:
        if layer.name == layer:
            set_trainable = True

        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    return conv_base

def VGG16_mod(input_shape, input_dim, train_generator, val_generator, freezed=False):
    conv_base = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=input_shape)
    
    if freezed == True:
        conv_base = freeze(conv_base)

    #conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dense(1, activation='sigmoid'))


    model.compile(optimizer=RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])

    history = model.fit_generator(train_generator,
                                steps_per_epoch=100,
                                epochs=30,
                                validation_data=val_generator,
                                validation_steps=50)
    
    return history, model

def simple_CNN(img_shape, n_labels, learning_rate, multiclass=None):
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=img_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=n_labels, activation='sigmoid'))

    if multiclass:
        loss = CategoricalCrossentropy()

    model.compile(loss = loss,
                optimizer=Adam(learning_rate=learning_rate),
                metrics=['acc'])
    
    return model