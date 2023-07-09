import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_img_from_directory(path, n_labels, n_img_label):
    total_img = list()

    plt.figure(figsize=(n_labels, 4))

    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if os.path.isdir(label_path):
            list_img = os.listdir(label_path)[:n_img_label]

            img_list = [os.path.join(label_path, img_path) for img_path in list_img]
            total_img.extend(img_list)
        else:
            print(f'{label_path} is not a folder')

    for idx, img in enumerate(total_img):
        subplot = plt.subplot(n_labels, 3, idx+1)
        subplot.axis('Off')

        img = mpimg.imread(img)
        plt.imshow(img)

def acc_loss_plots(history):
    acc = history.history()['acc']
    val_acc = history.history()['val_acc']

    loss = history.history()['loss']
    val_loss = history.history()['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def smooth_curve(points, factor=0.8):
    smothed_points = list()
    for point in points:
        if smothed_points:
            previous = smothed_points[-1]
            smothed_points.append(previous*factor+point*(1-factor))
        else:
            smothed_points.append(point)
    
    return smothed_points

