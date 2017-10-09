import numpy as np
from matplotlib import pyplot as plt


def show_slices(array, filename=None, every=5, cols=10, figsize=(24, 12)):
    """Plot z-axis slices of the specified 3D array.

    Args:
        array (numpy.ndarray): Array to plot.
        filename (str): Path to save image. If None, no image is saved.
        every (int): To print all slices set this value to 1.
        cols (int): Number of columns in the figure.
        figsize (tuple): Figure size.
    """
    
    n = int(np.ceil(len(array)/every))
    rows = int(np.ceil(n/cols))

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i, idx in enumerate(range(0, len(array), every)):
        r = i // cols
        c = i % cols
        
        if rows == 1:
            axes = ax[c]
        elif cols == 1:
            axes = ax[r]
        else:
            axes = ax[r, c]
        
        axes.set_title('slice %d' % idx)
        axes.imshow(array[idx], cmap='gray')
        axes.axis('off')
    
    if filename:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close('all')
    
    
    
