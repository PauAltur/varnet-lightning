import torch
import torch.nn as nn


class FFT2D(nn.Module):
    """
    Class that implements a 2D FFT transform.

    """

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the 2D FFT to the input data.

        Parameters
        ----------
        data : torch.Tensor
            The input data whose last two dimensions
            have shape (height, width).
        """
        data = torch.fft.fftn(data, dim=(-2, -1))
        data = torch.fft.fftshift(data, dim=(-2, -1))
        return data
