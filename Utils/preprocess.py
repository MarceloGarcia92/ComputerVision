from evaluate import deprocess_image

import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, image_utils
from keras.applications.vgg16 import preprocess_input, decode_predictions

def img_preprocess(img_path, target_size=(224, 224)):
    img = image_utils.load_img(img_path, target_size=target_size)
    x = image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def read_data_folder(folder: str):
    base_dir = f'{os.getcwd()}/{folder}'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    validation_dir = os.path.join(base_dir, 'validation')

    return train_dir, test_dir, validation_dir


import pandas as pd
import os

from sklearn.model_selection import train_test_split

def create_train_df(folder):
    filenames = os.listdir(f'{os.getcwd()}/{folder}/train')
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        corr = filename.split('.')[1]
        if category == 'dog':
            categories.append("dog")
        else:
            categories.append("cat")
        
        os.rename(f'{os.getcwd()}/{folder}/train/{filename}', f'{os.getcwd()}/{folder}/train/{category}{corr}.')

    filenames = os.listdir(f'{os.getcwd()}/{folder}/train')

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = val_df.reset_index(drop=True)

    return train_df, validate_df

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


def generator_creation(df, folder: str):
    generator = train_datagen.flow_from_dataframe(df, 
                                                folder,
                                                x_col='filename',
                                                y_col='category',  
                                                target_size=(150, 150),
                                                batch_size=20,
                                                class_mode='binary')
    
    return generator