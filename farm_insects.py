import os
import wget
from keras import layers
from keras import Model

from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from Utils.plots import show_img_from_directory

PATH = 'Data/farm_insects'
TARGET_SIZE = (150, 150, 3)

labels = [label for label in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, label))]
n_labels = len(labels)

#show_img_from_directory(path=PATH, n_labels=n_labels, n_img_label=3)



local_weights = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=TARGET_SIZE,
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

own_layers = layers.Flatten()(last_output)
own_layers = layers.Dense(1024, activation='relu')(own_layers)
own_layers = layers.Dropout(0.2)(own_layers)  
own_layers = layers.Dense(n_labels, activation='sigmoid')(own_layers)


model = Model(pre_trained_model.input, own_layers)

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc'])

#Creation of Generators with Image Agmentation technics
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest', 
                               validation_split=0.2)

#Creation of the generator from directory
train_generator = train_gen.flow_from_directory(
    PATH,
    target_size=TARGET_SIZE[:2],  
    batch_size=20,
    subset='training',
    class_mode='categorical')

val_generator = train_gen.flow_from_directory(
    PATH,
    target_size=TARGET_SIZE[:2],  
    batch_size=20,
    subset='validation',
    class_mode='categorical')

hist = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=val_generator,
    validation_steps=50
)

model.save('type_insects.h5')