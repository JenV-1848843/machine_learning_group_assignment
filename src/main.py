#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Jens Sanen, Wout Bosteels, Jen Verboven"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        # The final layer should be of size one as we're predicting one output value (close stock value)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        # _ here is the updated (h0, c0) tuple which we don't care about in this case
        out, _ = self.lstm(x, (h0, c0))
        # transform the output from the fc layer to the appropriate size
        out = self.fc(out[:, -1, :])
        return out

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Last Close(t-{i})'] = df['Last Close'].shift(i)

    df.dropna(inplace=True)

    return df

def normalize(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(df)

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

    lookback = 7
    # Adds lookback columns to the dataframe (the values of last close from the previous day up until "lookback" days backwards)
    shifted_training_df = prepare_dataframe_for_lstm(train_data, lookback)
    shifted_testing_df = prepare_dataframe_for_lstm(test_data, lookback)

    shifted_training_df_as_np = shifted_training_df.to_numpy()
    shifted_testing_df_as_np = shifted_testing_df.to_numpy()

    shifted_training_df_as_np = normalize(shifted_training_df_as_np)
    shifted_testing_df_as_np = normalize(shifted_testing_df_as_np)

    # Take all columns except for the actual last close column as X (so open, high, low, and the lookback values of last close)
    X_train = np.concatenate((shifted_training_df_as_np[:, :3], shifted_training_df_as_np[:, 4:]), axis=1)
    X_test = np.concatenate((shifted_testing_df_as_np[:, :3], shifted_testing_df_as_np[:, 4:]), axis=1)
    # Take the actual last close column as the feature to predict
    y_train = shifted_training_df_as_np[:, 3]
    y_test = shifted_testing_df_as_np[:, 3]
    # flip the training feature dataframe so that the oldest of the lookback values is at the front of the dataframe so the LSTM model can learn the trend going forwards in time
    X_train = dc(np.flip(X_train, axis=1))

    # Add an extra dimension needed for the Pytorch LSTM model
    X_train = X_train.reshape((-1, X_train.shape[1], 1))
    X_test = X_test.reshape((-1, X_test.shape[1], 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # Wrap numpy dataframes in Pytorch tensors and make all values floats
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    # Make the Pytorch tensors into datasets for use with Pytorch
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Wrap the datasets in data loaders to make batches
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Helper code for visualization
    # for _, batch in enumerate(train_loader):
    #     X_batch, y_batch = batch[0].to(device), batch[1].to(device)
    #     print(X_batch.shape, y_batch.shape)
    #     break

    model = LSTM(1, 10, 1)
    model.to(device)

    learning_rate = 0.001
    num_epochs = 50
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_one_epoch():
        model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(X_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # Step in direction of the gradient
            optimizer.step()

            if batch_index % 100 == 99: #print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+0, avg_loss_across_batches))
                running_loss = 0.0
        print()

    def validate_one_epoch():
        model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(X_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()

    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()

    # Code for plotting predictions vs actual
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

    plt.plot(y_train, label = 'Actual Last Close')
    plt.plot(predicted, label = 'Predicted Last Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    # plt.plot(train_data['Date'], train_data['Last Close'])
    # plt.show()
    # train_values = train_data['Last Close'].values.reshape(-1, 1)
    # test_values = test_data['Last Close'].values.reshape(-1, 1)

    # plot_data(train_data)

    # train_values = normalize(train_values)

    # seq_length = 2
    # x, y = create_sequences(train_values, seq_length)


    # TODO --> replace these test arrays with the actual values and predicted values
    # This is solely to demonstrate how to use the mean_absolute_percentage_error function
    # y_true = [3, -0.5, 2, 7]
    # y_pred = [2.5, 0.0, 2, 8]
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    # print("MAPE: {}".format(mape))

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