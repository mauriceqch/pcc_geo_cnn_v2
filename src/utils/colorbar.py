import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_colorbar(a_min, a_max, cmap='inferno', orientation='horizontal', figsize=None):
    assert orientation in ('horizontal', 'vertical')
    a = np.array([[a_min, a_max]])
    if figsize is None:
        figsize = (6, 0.5) if orientation == 'horizontal' else (1.0, 6)
    fig = plt.figure(figsize=figsize)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    if orientation == 'horizontal':
        cax = plt.axes([0.05, 0.5, 0.9, 0.2])
    else:
        cax = plt.axes([0.2, 0.05, 0.15, 0.9])
    plt.colorbar(orientation=orientation, cax=cax)
    norm = mpl.colors.Normalize(vmin=a_min, vmax=a_max)
    return fig, lambda x: cmap(norm(x))


if __name__ == '__main__':
    fig, cmap = get_colorbar(0, 128, orientation='horizontal')
    plt.show()
    fig, cmap = get_colorbar(0, 128, orientation='vertical')
    plt.show()
    x = [0, 64, 128, 192, 256]
    print(x, cmap(x))
