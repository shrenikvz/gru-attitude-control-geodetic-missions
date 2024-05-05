# Leveraging Gated Recurrent Units for Iterative Online Precise Attitude Control for Geodesic Missions

This repository replicates the result of the paper (https://arxiv.com).

## Code outline

The code is structured as follows:

+ [src](./src): Contains the Python code that implements the methodology.
+ [examples](./examples): Contains the Python code that replicates the results of the paper, and the Python scripts that make the plots.
examples, and the Python scripts that make the plots.

## Installing the code

The requirements for the Python plotting scripts are (ignoring standard libraries):
+ [matplotlib](https://matplotlib.org)
+ [seaborn](https://seaborn.pydata.org)
+ [texlive](https://tug.org/texlive/) If you are on OS X and using homebrew, then run
    then run
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

**Note:** Because the random seed changes every time you run the code, the figures you produce may be slightly different from the ones found in the paper. We decided not to fix the random seed to avoid a positive results bias, or ``hero'' runs.


