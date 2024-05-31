from setuptools import find_packages, setup

setup(
    name="projectB",
    version="0.1.0",
    description=(
        "My attempt at completing project B as an"
        " assignment for the course 'Model and"
        " learning-based inverse problems in imaging'"
        " at ETH Zurich."
    ),
    author="Pau Altur Pastor",
    author_email="paltur@student.ethz.ch",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
