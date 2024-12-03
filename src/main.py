#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Jens Sanen, Wout Bosteels, Jen Verboven"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def normalize(values):
    scaler = MinMaxScaler()
    values = scaler.fit_transform(values)
    return values

def plot_data(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Last Close'], label="Last Close")
    plt.title("Last close prices over time")
    plt.xlabel('Date')
    plt.ylabel('Last Close')
    plt.legend()
    plt.grid()
    plt.show()

def read_files(train_file, test_file):
    df1 = pd.read_csv(train_file)
    df2 = pd.read_csv(test_file)
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])

    return df1, df2

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.FloatTensor(x), torch.FloatTensor(y)

def main(args):
    """ Main entry point of the app """
    training_file = args.training_file
    testing_file = args.testing_file
    # HERE GOES YOUR CODE TO CALCULATE THE MAPE
    # FEEL FREE TO IMPLEMENT HELPER FUNCTIONS

    train_data, test_data = read_files(training_file, testing_file)
    train_values = train_data['Last Close'].values.reshape(-1, 1)
    test_values = test_data['Last Close'].values.reshape(-1, 1)

    train_values = normalize(train_values)

    seq_length = 2
    # x, y = create_sequences(train_values, seq_length)


    # TODO --> replace these test arrays with the actual values and predicted values
    # This is solely to demonstrate how to use the mean_absolute_percentage_error function
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("MAPE: {}".format(mape))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
