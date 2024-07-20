from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def map_to_range(array: np.array, out_min, out_max) -> np.array:
    return (array - np.min(array)) / (np.max(array) - np.min(array)) * (out_max - out_min) + out_min


def plot_loss(arrays: List[Tuple[np.array, str]], normalize: bool = False):
    if normalize:
        arrays = [(map_to_range(array, 0, 1), name) for array, name in arrays]

    plt.figure()
    for array, name in arrays:
        plt.plot(array, label=name)
    plt.title('Normalized Validation Loss History' if normalize else 'Validation History Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized MAE' if normalize else 'MAE')
    plt.legend()
    plt.show()


def main():
    arrays = [
        (np.array([0.09019, 0.07066, 0.07037, 0.06841, 0.06854, 0.06724, 0.06845, 0.0681, 0.06856]), 'Wheelchair only'),
        (np.array([9.984, 6.319, 3.734, 2.887, 2.833, 2.805, 2.745, 2.696, 2.668]), 'Car only'),
        (np.array([6.877, 6.85, 6.839, 6.835, 6.838, 6.838, 6.837, 6.836]), 'All classes')
    ]
    plot_loss(arrays, normalize=True)


if __name__ == '__main__':
    main()
