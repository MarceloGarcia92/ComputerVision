import matplotlib.pyplot as plt

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

