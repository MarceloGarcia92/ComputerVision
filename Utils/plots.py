import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_img_from_directory(path, n_labels, n_img=8):
    total_img = list()

    n = int(n_img/n_labels)

    plt.figure(figsize=(n, n))

    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        print(label_path)
        list_img = os.listdir(label_path)[:n]

        img_list = [os.path.join(label_path, img_path) for img_path in list_img]
        total_img.extend(img_list)

    for idx, img in enumerate(total_img):
        subplot = plt.subplot(n, n, idx+1)
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

