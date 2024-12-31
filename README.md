# Machine Learning Group Assignment

This repository contains the source code for the Machine Learning assignment by Bosteels Wout, Sanen Jens, and Verboven Jen.

## Setup

Before running the project, install the required dependencies by running the following command:

`pip install -r .\requirements.txt`


## Running the Model

There are two ways to run the model:

### 1. Train and Run a Single Model

To train and run the model once, follow these steps:

- Uncomment the section labeled `# ONE MODEL TRAINING` in the `main` function in `src/main.py`.
- This will train the model, evaluate it using the test dataset, and output the MAPE (Mean Absolute Percentage Error). It will also display a plot comparing predicted vs. actual values.

### 2. Train and Run Multiple Models

To train the same model multiple times, follow these steps:

- Uncomment the section labeled `# MULTIPLE MODELS TRAINING` in the `main` function in `src/main.py`.
- This will train and run the model with the same input multiple times (the number of iterations `n` can be set in this section). It will output the average, best, and worst MAPE values. Additionally, it will plot the MAPE for each model and compare the predicted vs. actual closing price for both the best and worst model.

## Running the Project

To run the project, provide the paths to the training and test CSV files as command-line arguments:

`python .\src\main.py <path to training data file> <path to testing data file>`


Ensure the CSV files are properly formatted as expected by the model.

