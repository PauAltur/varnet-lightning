# VarNet Lightning

This repository implements a Variational Network model for MRI reconstruction in PyTorch Lightning. The model is based on the paper [Learning a Variational Network for Reconstruction of Accelerated MRI Data](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26977) by Hammernik et al.

## Getting Started
This project has been implemented in Python 3.12.1, PyTorch 2.3.0, CUDA 12.1, and PyTorch Lightning 2.2.5. In Windows, I recommend using [pyenv-win](https://pyenv-win.github.io/pyenv-win/docs/installation.html) to install the appropriate python version, and venv to keep the development environment separate from other projects. To get started, create a virtual environment and install the required dependencies using the following command:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```