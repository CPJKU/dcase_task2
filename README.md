# Training General-Purpose Audio Tagging Networks with Noisy Labels and Iterative Self-Verification

This repository contains the corresponding code for the 2nd place submission
to the first [Freesound general-purpose audio tagging challenge](http://dcase.community/challenge2018/task-general-purpose-audio-tagging)
carried out as Task 2 within the [DCASE challenge 2018](http://dcase.community/challenge2018/).

For a **detailed description of the entire audio tagging system**
please visit the [corresponding github page](https://cpjku.github.io/dcase_task2/).
In this README I just provide the technical instructions to set up the project.


# Getting Started
Before we can start working with the code, we first need to set up a few things:

## Setup and Requirements
For a list of required python packages see the *requirements.txt*
or just install them all at once using pip.
```
pip install -r requirements.txt
```

To install the project in develop mode run
```
python setup.py develop --user
```
in the root folder of the package.

This is what I recommend, especially if you want to try out new ideas.


## Getting the Data
Then download the [challenge data](https://www.kaggle.com/c/freesound-audio-tagging/data) and organize it in the following folder structure:
```
<DATA_ROOT>
    - audio_train
    - audio_test
    - train.csv
    - test_post_competition.csv
```

## Set Data and Model path
In *config/settings.py* you have to set the following two paths:
```
DATA_ROOT = "/home/matthias/shared/datasets/dcase2018_task2_release"
EXP_ROOT = "/home/matthias/experiments/dcase_task2/"
```

*DATA_ROOT* is the *<DATA_ROOT>* path from above.<br>
*EXP_ROOT* is where the model parameters and logs will be stored.

Once this is all set up, you can switch to the detailed writeup on this [github page](https://cpjku.github.io/dcase_task2/).
