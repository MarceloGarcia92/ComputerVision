from preprocess import img_preprocess

import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras import models
from keras.applications import VGG16
from keras import backend as K


def acttivation_layers(model, img_tensor):
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    return activations

def eval_channel_inter_act(model, activations):
    layer_names = [layer.name for layer in model.layers[:8]]
    
    img_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[-1]
        n_cols = n_features // img_per_row
        display_grid = np.zeros((size * n_cols, img_per_row *size))

        for col in range(n_cols):
            for row in range(img_per_row):
                channel_img = layer_activation[0,
                                               :, :,
                                               col * img_per_row + row]
                
                channel_img -= channel_img.mean()
                channel_img /= channel_img.std()
                channel_img *= 64
                channel_img += 128
                channel_img = np.clip(channel_img, 0, 255).astype('unit8')
                display_grid[col*size : (col+1)*size,
                             row*size : (row+1)*size] = channel_img
                
            scale = 1./size
            plt.figure(figsize=(scale*display_grid.shape[1],
                                scale*display_grid.shape[1]))
            
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')


model = VGG16(weights='imagenet',
              include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    #Normalization
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.rand((1, size, size, 3))*20 + 128

    step=1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value*step

    img = input_img_data[0]
    return deprocess_image(img)


def grid_filter_conv(layer_name, size=64, margin=5):
    result = np.zeros((8*size + 7 * margin, 8*size + 7 *margin, 3))

    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name, i+(j*8), size=size)

            horizontal_start=i*size+i*margin
            horizontal_end=horizontal_start+size
            vertical_start=j*size*margin
            vertical_end=vertical_start+size
            result[horizontal_start:horizontal_end,
                    vertical_start:vertical_end, :] = filter_img

    plt.figure(figsize=(20,20))
    plt.imshow(result)


def grad_CAM(class_idx:int, layer:str, img_path:str):
    x = img_preprocess(img_path)

    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer(layer)

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)

    return heatmap

def superimposing_heatmap_img(img_path, heatmap):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap*0.4+img

    plt.matshow(superimposed_img)
