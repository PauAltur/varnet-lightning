import numpy as np
import numpy.typing as npt


class UniformUndersampler:
    """
    Class that implements a uniform undersampling transform for 2D data.

    """

    def __init__(self, factor: float, hw_center: int, seed: int):
        self.factor = factor
        self.hw_center = hw_center
        self.seed = seed

    def forward(self, data: npt.NDArray) -> npt.NDArray:
        n_lines = np.prod(data.shape[:-1])
        original_shape = data.shape

        center_line = n_lines // 2
        center_lines = np.where(np.arange(n_lines) % center_line == 0)[0][1::2]
        hw_center_arr = np.arange(-self.hw_center, self.hw_center + 1)
        unselectable_lines = np.concatenate(center_lines[:, None] + hw_center_arr)
        selectable_lines = np.setdiff1d(np.arange(n_lines), unselectable_lines)
        selected_lines = np.random.RandomState(self.seed).choice(
            selectable_lines, int(self.factor * n_lines), replace=False
        )
        data = data.reshape(-1, data.shape[-1])
        data[~selected_lines] = 1
        data = data.reshape(original_shape)

        return data
