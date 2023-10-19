#!/usr/bin/env python
# coding:utf-8

import os
from itertools import cycle

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

plt.rc('font',family='Arial', size=11)


def plot_roc(save_path, params, task, n_classes=3):
    """
    """
    fpr, tpr, roc_auc = params['fpr'], params['tpr'], params['roc_auc']
    if task == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.4f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.4f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                    label='ROC curve of class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))
    else:
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % (roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck', lw=1)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The ROC curve of node-{}'.format(task))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, "ROC curve of nodel-{}.png".format(task)),dpi=400)
    plt.close()


def plot_prc(save_path, params, task, n_classes=3):
    """
    """
    precision, recall, auprc = params['precision'], params['recall'], params['auprc']
    if task == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.step(recall["micro"], 
             precision["micro"], 
             alpha=1, 
             lw=1, 
             color="deeppink",
             where='post', 
             label= f"micro-average AUPRC curve (area = {auprc['micro']:.4f})")
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.step(recall[i], 
                     precision[i], 
                     alpha=1, 
                     lw=1,
                     color=color, 
                     where='post', 
                     label= "AUPRC curve of class {0} (area = {1:0.4f})".format(i, auprc[i]))
    else:
        plt.step(recall, 
                precision, 
                alpha=1, 
                lw=1, 
                where='post', 
                label='AUPRC (area = %0.4f)' % (auprc))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('The PRC curve of node-{}'.format(task))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, "PRC curve of node-{}.png".format(task)),dpi=400)
    plt.close()


def plot_cm(save_path, y_true, y_pred, task):
    # get the confusion matrix
    C = confusion_matrix(y_true, y_pred)
    if task == 1:
        fig, ax = plt.subplots(figsize = (3, 3))
    else:
        fig, ax = plt.subplots(figsize = (2, 2))
    ax.matshow(C, cmap = plt.cm.Blues, alpha = 0.6)
    for n in range(C.shape[0]):
        for m in range(C.shape[1]):
            ax.text(x = m, y = n,
                s = C[n, m], 
                va = 'center', 
                ha = 'center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(save_path, "confusion matrix of node-{}.png".format(task)),dpi=400)
    plt.show()
    plt.close()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation=-30,
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for key, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts