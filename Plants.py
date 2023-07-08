import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from Utils.models import simple_CNN


TRAIN_PATH = f'{os.getcwd()}/Data/plant-seedlings-classification/train'
EXAMPLE_PATH = f'{TRAIN_PATH}/Sugar beet/6d579671c.png'
LEARNING_RATE = 0.001

print('Setting data parameters')
#Reading one image example
img = Image.open(EXAMPLE_PATH)
arr = np.asarray(img)
img_shape = arr.shape
print(f'Shape of input image {img_shape}')

#Setting number of labels on the data
n_labels = len([folder for folder in os.listdir(TRAIN_PATH) if os.path.isdir(f'{TRAIN_PATH}/{folder}')])

print(f'Shape of imgs: {img_shape}')
print(f'Number of labels: {n_labels}')

print('*'*100)
print('Summary model')

model = simple_CNN(img_shape, n_labels, LEARNING_RATE, multiclass=True)

print(model.summary())
print('*'*100)

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    shuffle=True,
    target_size=img_shape[:2],
    batch_size=128,
    class_mode='categorical', #binary
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    shuffle=True,
    target_size=img_shape[:2],
    batch_size=128,
    class_mode='categorical', #binary
    subset='validation'
)

print('Model training')

hist = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=1,
    validation_data=val_generator,
    validation_steps=8,
    batch_size=32
)

model.save('models/plants_model.h5')