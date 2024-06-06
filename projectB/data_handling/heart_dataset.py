import logging
import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from projectB.utils.fft import numpy_2_complex
from projectB.utils.mri import (
    dummy_coil_sensitivities,
    mriAdjointOp,
    mriForwardOp,
    uniform_undersampling_mask,
)

# TODO: This should really be a dataloader rather than a dataset

DEFAULT_OPTS = {
    "root_dir": "..\\..\\data\\raw",
    "name": "2dt_heart.mat",
    "factor": 0.25,
    "hw_center": 4,
    "seed": 42,
    "flatten": True,
    "normalization": "max",
    "mode": "train",
    "training_set": list(range(325)),
    "testing_set": list(range(325, 375)),
}


class HeartDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, **kwargs):
        self.options = DEFAULT_OPTS
        # any options passed through the constructor will override the default options
        for key in kwargs.keys():
            self.options[key] = kwargs[key]

        data_path = os.path.join(self.options["root_dir"], self.options["name"])
        msg = f"No file found with path: {data_path}"
        assert os.path.exists(data_path), msg

        self.load_data(data_path)

    def load_data(self, data_path):
        """_summary_

        Args:
            data_path (_type_): _description_
        """
        # load images and swap axes so shape is (N, T, H, W)
        data = loadmat(data_path)
        self.ref = data["imgs"].swapaxes(0, 3).swapaxes(1, 2)
        if self.options["mode"] == "train":
            self.ref = self.ref[self.options["training_set"]]
        elif self.options["mode"] == "test":
            self.ref = self.ref[self.options["testing_set"]]
        else:
            raise ValueError("mode must be either 'train' or 'test'")

        # generate masks
        self.masks = uniform_undersampling_mask(
            self.ref,
            factor=self.options["factor"],
            hw_center=self.options["hw_center"],
            seed=self.options["seed"],
        )

        # generate dummy coil sensitivities
        self.coil_sens = dummy_coil_sensitivities(self.ref.shape)

        # generate masked Fourier transform of images
        self.f = mriForwardOp(self.ref, self.coil_sens, self.masks)

        # generate initial reconstruction guess
        self.inputs = mriAdjointOp(self.f, self.coil_sens, self.masks)

        # reshape to add coil dimension
        self.f = self.f[:, :, None, ...]
        self.coil_sens = self.coil_sens[:, :, None, ...]

        # normalize by max of reconstruction guess
        if "normalization" in self.options and self.options["normalization"] == "max":
            norm = np.max(np.abs(self.inputs))
        else:
            logging.Warning("No normalization applied to the data")
            norm = 1.0
            self.options["normalization"] = None

        self.inputs = self.inputs / norm
        self.ref = self.ref / norm
        self.coil_sens = self.coil_sens / norm

        if "flatten" in self.options and self.options["flatten"] is True:
            self.inputs = self.inputs.reshape(
                self.inputs.shape[0] * self.inputs.shape[1], *self.inputs.shape[2:]
            )
            self.ref = self.ref.reshape(
                self.ref.shape[0] * self.ref.shape[1], *self.ref.shape[2:]
            )
            self.coil_sens = self.coil_sens.reshape(
                self.coil_sens.shape[0] * self.coil_sens.shape[1],
                *self.coil_sens.shape[2:],
            )
            self.masks = self.masks.reshape(
                self.masks.shape[0] * self.masks.shape[1], *self.masks.shape[2:]
            )
            self.f = self.f.reshape(
                self.f.shape[0] * self.f.shape[1], *self.f.shape[2:]
            )

        # transform complex numpy arrays into torch tensors with 2 channels {Re, Im}
        self.inputs = numpy_2_complex(self.inputs)
        self.f = numpy_2_complex(self.f)
        self.coil_sens = numpy_2_complex(self.coil_sens)
        self.ref = numpy_2_complex(self.ref)
        self.masks = torch.tensor(self.masks).float()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return {
            "u_t": self.inputs[idx],
            "f": self.f[idx],
            "coil_sens": self.coil_sens[idx],
            "sampling_mask": self.masks[idx],
            "reference": self.ref[idx],
        }


if __name__ == "__main__":
    dataset = HeartDataset()
    print(len(dataset))
    print(dataset.inputs.shape)
    print(dataset.f.shape)
    print(dataset.coil_sens.shape)
    print(dataset.masks.shape)
    print(dataset.ref.shape)
