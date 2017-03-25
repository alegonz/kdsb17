import numpy as np
from matplotlib import pyplot as plt


def show_slices(array, filename=None, every=5, cols=10):
    
    n = int(np.ceil(len(array)/every))
    rows = int(np.ceil(n/cols))
    
    fig, ax = plt.subplots(rows, cols, figsize=[24, 12])
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
    
    
    
