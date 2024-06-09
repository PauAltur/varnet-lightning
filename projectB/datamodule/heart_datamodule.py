import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

from projectB.utils.fft import numpy_2_complex
from projectB.utils.mri import (
    dummy_coil_sensitivities,
    mriAdjointOp,
    mriForwardOp,
    uniform_undersampling_mask,
)


class HeartDatamodule(pl.LightningDataModule):
    """Data module for the heart dataset. This dataset
    contains 2D+t cardiac MRI data. The data is loaded from a .mat file.
    Only the reconstructed images are available. The undersampled K-space
    data is generated by applying a Fourier transform and a k-space mask
    to the reconstructed images. The coil sensitivities are generated
    by creating a dummy coil sensitivity map. The initial guess of the
    reconstructed images is obtained by applying the adjoint operator to
    the undersampled K-space data. The data is split into training,
    validation, and test sets.
    """

    def __init__(
        self,
        data_dir: str,
        name: str,
        factor: float,
        hw_center: int,
        flatten: bool,
        normalize: bool,
        train_indices: list,
        val_indices: list,
        test_indices: list,
        batch_size: int,
    ):
        """Constructor method.

        Parameters
        ----------
        data_dir : str
            Directory of the data file.
        name : str
            Name of the data file.
        factor : float
            Undersampling factor of the k-space mask.
        hw_center : int
            Half-width of the center of the k-space mask. This will always
            be sampled.
        flatten : bool
            Whether to flatten the data. This removes the time dimension
            and treats each timestep as a separate sample.
        normalize : bool
            Whether to normalize the data by the maximum of the initial
            guess of the reconstructed images.
        train_indices : list
            The indices of the samples to use for the training set.
        val_indices : list
            The indices of the samples to use for the validation set.
        test_indices : list
            The indices of the samples to use for the test set.
        batch_size : int
            The batch size to use for the training dataloader.
        data : dict
            The data loaded from the specified path.
        ref : torch.tensor
            The ground truth images. It has a shape of (num_samples, height, width, 2).
        masks : torch.tensor
            The undersampling masks of the images. It has a shape of
            (num_samples, height, width).
        coil_sens : torch.tensor
            The coil sensitivities of the images. It has a shape of
            (num_samples, num_coils, height, width, 2) or (num_samples,
            num_timesteps, num_coils, height, width, 2)
        f : torch.tensor
            The undersampled K-space of the images, obtained through
            FFT and masking. It has a shape of (num_samples, num_coils,
            height, width, 2) or (num_samples, num_timesteps, num_coils,
            height, width, 2).
        inputs : torch.tensor
            The initial guess of the reconstructed images.
            It has a shape of (num_samples, height, width, 2)
            or (num_samples, num_timesteps, height, width, 2).
        train_dataset : TensorDataset
            Training dataset.
        val_dataset : TensorDataset
            Validation dataset.
        test_dataset : TensorDataset
            Test dataset.
        """
        super().__init__()
        self.data_dir = data_dir
        self.name = name
        self.factor = factor
        self.hw_center = hw_center
        self.flatten = flatten
        self.normalize = normalize
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.batch_size = batch_size

    def setup(self):
        """Set up method of the datamodule."""
        self._load_data()
        self._compose_dataset()
        self._preprocess_dataset()
        self._split_dataset()

    def _load_data(self):
        """Loads the data from the specified path."""
        datapath = os.path.join(self.data_dir, self.name)
        msg = f"No file found with path: {datapath}"
        assert os.path.exists(datapath), msg

        print(f"Loading data from {datapath}...")
        self.data = loadmat(datapath)

    def _compose_dataset(self):
        """Composes the dataset by generating the undersampling masks,
        coil sensitivities, undersampled K-space data and initial guesses
        of the reconstruced images.
        """
        print("Composing dataset...")
        self.ref = self.data["imgs"].swapaxes(0, 3).swapaxes(1, 2)
        self.masks = uniform_undersampling_mask(
            self.ref,
            factor=self.factor,
            hw_center=self.hw_center,
        )
        self.coil_sens = dummy_coil_sensitivities(self.ref.shape)
        self.f = mriForwardOp(self.ref, self.coil_sens, self.masks)
        self.inputs = mriAdjointOp(self.f, self.coil_sens, self.masks)

    def _preprocess_dataset(self):
        """Preprocesses the dataset by (optionally) flattening it, normalizing it,
        adding a coil dimension to the Fourier transform and coil sensitivities, and
        converting the numpy arrays to complex tensors.
        """
        # TODO: refactor this to use transforms
        print("Preprocessing dataset...")
        if self.flatten:
            coil_axis = 1
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
        else:
            coil_axis = 2

        self.f = np.expand_dims(self.f, axis=coil_axis)
        self.coil_sens = np.expand_dims(self.coil_sens, axis=coil_axis)

        if self.normalize:
            norm = np.max(np.abs(self.inputs))
        else:
            logging.warning("No normalization applied to the data")
            norm = 1.0

        self.inputs = self.inputs / norm
        self.ref = self.ref / norm
        self.coil_sens = self.coil_sens / norm

        self.inputs = numpy_2_complex(self.inputs)
        self.f = numpy_2_complex(self.f)
        self.coil_sens = numpy_2_complex(self.coil_sens)
        self.ref = numpy_2_complex(self.ref)
        self.masks = torch.tensor(self.masks)

    def _split_dataset(self):
        """Splits the dataset into training, validation, and test sets.
        The samples are defined by the first axis.
        """
        print("Splitting dataset...")
        self.train_dataset = TensorDataset(
            self.inputs[self.train_indices],
            self.f[self.train_indices],
            self.coil_sens[self.train_indices],
            self.masks[self.train_indices],
            self.ref[self.train_indices],
        )

        self.val_dataset = TensorDataset(
            self.inputs[self.val_indices],
            self.f[self.val_indices],
            self.coil_sens[self.val_indices],
            self.masks[self.val_indices],
            self.ref[self.val_indices],
        )

        self.test_dataset = TensorDataset(
            self.inputs[self.test_indices],
            self.f[self.test_indices],
            self.coil_sens[self.test_indices],
            self.masks[self.test_indices],
            self.ref[self.test_indices],
        )

    def train_dataloader(self):
        """Instantiates the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Instantiates the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.
        """
        return DataLoader(
            self.val_dataset, batch_size=len(self.val_dataset), shuffle=False
        )

    def test_dataloader(self):
        """Instantiates the test dataloader.

        Returns
        -------
        DataLoader
            The test dataloader.
        """
        return DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset), shuffle=False
        )


if __name__ == "__main__":
    datamodule = HeartDatamodule(
        data_dir="data\\raw",
        name="2dt_heart.mat",
        factor=0.25,
        hw_center=2,
        flatten=True,
        normalize=True,
        train_indices=np.arange(3250),
        val_indices=np.arange(3250, 3750),
        test_indices=np.arange(3750, 4125),
        batch_size=2,
    )
    datamodule.setup()
    train_set = datamodule.train_dataset
    val_test = datamodule.val_dataset
    test_set = datamodule.test_dataset

    print("Train set:", len(train_set))
    print("-----------------------------------------------------------------")
    print("Inputs: ", train_set[0][0].shape)
    print("F: ", train_set[0][1].shape)
    print("Coil sensitivities: ", train_set[0][2].shape)
    print("Masks: ", train_set[0][3].shape)
    print("Reference: ", train_set[0][4].shape)
    print("\n")

    print("Validation set:", len(val_test))
    print("-----------------------------------------------------------------")
    print("Inputs: ", val_test[0][0].shape)
    print("F: ", val_test[0][1].shape)
    print("Coil sensitivities: ", val_test[0][2].shape)
    print("Masks: ", val_test[0][3].shape)
    print("Reference: ", val_test[0][4].shape)
    print("\n")

    print("Test set:", len(test_set))
    print("-----------------------------------------------------------------")
    print("Inputs: ", test_set[0][0].shape)
    print("F: ", test_set[0][1].shape)
    print("Coil sensitivities: ", test_set[0][2].shape)
    print("Masks: ", test_set[0][3].shape)
    print("Reference: ", test_set[0][4].shape)
    print("\n")