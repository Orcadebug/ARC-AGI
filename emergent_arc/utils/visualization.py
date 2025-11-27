import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def plot_grid(grid: np.ndarray, title: str = ""):
    cmap = colors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid.shape[1], 1))
    plt.yticks(np.arange(-0.5, grid.shape[0], 1))
    plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    if title:
        plt.title(title)
    plt.show()

def plot_task(task: dict):
    train_pairs = task['train']
    test_pairs = task['test']
    
    n_train = len(train_pairs)
    n_test = len(test_pairs)
    
    fig, axes = plt.subplots(n_train + n_test, 2, figsize=(8, 4 * (n_train + n_test)))
    
    cmap = colors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    for i, (inp, out) in enumerate(train_pairs):
        axes[i, 0].imshow(inp, cmap=cmap, norm=norm)
        axes[i, 0].set_title(f"Train {i} Input")
        axes[i, 1].imshow(out, cmap=cmap, norm=norm)
        axes[i, 1].set_title(f"Train {i} Output")
        
    for i, pair in enumerate(test_pairs):
        inp = pair['input']
        out = pair.get('output', np.zeros_like(inp))
        row = n_train + i
        axes[row, 0].imshow(inp, cmap=cmap, norm=norm)
        axes[row, 0].set_title(f"Test {i} Input")
        axes[row, 1].imshow(out, cmap=cmap, norm=norm)
        axes[row, 1].set_title(f"Test {i} Output")
        
    plt.tight_layout()
    plt.show()
