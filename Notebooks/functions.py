import matplotlib.pyplot as plt


def plotter(history, metric,ax):
    '''used to plot metric for keras NNs'''
    ax.plot(history.history[metric])
    ax.plot(history.history['val_'+metric])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend([metric, 'val_'+metric])
    