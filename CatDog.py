import zipfile
import wget
import os

from keras.preprocessing.image import ImageDataGenerator

from Utils.plots import show_img_from_directory
from Utils.models import simple_CNN

#Setting global variables
TRAIN_PATH = 'Data/cats_and_dogs_filtered/train'
VAL_PATH = 'Data/cats_and_dogs_filtered/validation'
TARGET_SIZE = (150, 150, 3)

#Creation of variables
n_labels = len(os.listdir(TRAIN_PATH))

#Getting the data
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
filename = wget.download(url)
print(f'file {filename} saved')

#Extracting zip file into the filesystem
zip_ref = zipfile.ZipFile("./cats_and_dogs_filtered.zip", 'r')
zip_ref.extractall("Data/")
zip_ref.close()

#Showing some data from the folders
show_img_from_directory(TRAIN_PATH, n_labels=n_labels)


#Creation of Generators with Image Agmentation technics
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

val_gen = ImageDataGenerator(rescale=1./255)

#Creation of the generator from directory
train_generator = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=TARGET_SIZE[:2],  
    batch_size=20,
    class_mode='binary')

val_generator = val_gen.flow_from_directory(
    VAL_PATH,
    target_size=TARGET_SIZE[:2],  
    batch_size=20,
    class_mode='binary')

#Creation of the model by simple CNN
model = simple_CNN(n_labels=1, learning_rate=0.01, img_shape=TARGET_SIZE, binaryclass=True)

#Training the model
hist = model.fit(
      train_generator,
      steps_per_epoch=100, 
      epochs=2,
      validation_data=val_generator,
      validation_steps=50)

model.save('catVsDog.h5')