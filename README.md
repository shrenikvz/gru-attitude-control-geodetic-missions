# gru-attitude-control-geodetic-missions

This repository replicates the result of the paper "Leveraging Gated Recurrent Units for Iterative Online Precise Attitude Control for Geodetic Missions".

Authors: [Vrushabh Zinage](https://scholar.google.com/citations?user=hyHy9DQAAAAJ&hl=en&oi=ao), [Shrenik Zinage](https://scholar.google.com/citations?user=CuaVvXsAAAAJ&hl=en), [Srinivas Bettadpur](https://scholar.google.com/citations?user=u4E5AboAAAAJ&hl=en&oi=ao), [Efstathios Bakolas](https://scholar.google.com/citations?user=mnXfihQAAAAJ&hl=en&oi=ao).

## Code outline

The code is structured as follows:

+ [data](./data): Data for disturbances.
+ [src](./src): Contains the Python codes required to implement the methodology.
+ [figures](./figures): Contains saved figures.
+ [main](./main.py): Python code that implements the methodology. 

## Installation

Ensure that you have Python 3.8 or later installed on your system.
In addition to the common python packages (`numpy`, `matplotlib`, etc.), you will need to install [jax](https://jax.readthedocs.io/en/latest/index.html) to run the codes.

**Note:** If, while installing `jax` on a Mac with an M1 chip, you run into the error saying something like
```
RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support. You may be able work around this issue by building jaxlib from source.
```
a workaround for now is to install `jax` and `jaxlib` via conda-forge, as mentioned in [this comment](https://github.com/google/jax/issues/5501#issuecomment-1032891169).
```
conda install -c conda-forge jaxlib
conda install -c conda-forge jax
```

To install `equinox` and `optax` libraries type the following in the terminal:

```
pip install equinox
pip install optax
```

The requirements for the Python plotting scripts are (ignoring standard libraries):
+ [matplotlib](https://matplotlib.org)
+ [seaborn](https://seaborn.pydata.org)
+ [texlive](https://tug.org/texlive/) If you are on OS X and using homebrew, then run
    ```
    brew install texlive
    ```
    If for some reason you cannot install texlive, you will need to manually edit the Python plotting scripts and comment out the lines:
    ```
        plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
        params = {'text.usetex' : True,
                'font.size' : 9,
                'font.family' : 'lmodern'
                }
    ```

## Reproducing the paper results

**Note:** Due to the variability of the random seed with each execution of the code, the resulting figures may differ slightly from those presented in the paper. We chose not to fix the random seed to avoid a positive results bias, or ``hero'' runs.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{zinage2024leveraging,
  title={Leveraging Gated Recurrent Units for Iterative Online Precise Attitude Control for Geodetic Missions},
  author={Zinage, Vrushabh and Zinage, Shrenik and Bettadpur, Srinivas and Bakolas, Efstathios},
  journal={arXiv preprint arXiv:2405.15159},
  year={2024}
}
