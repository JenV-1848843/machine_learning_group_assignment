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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from copy import deepcopy as dc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

OG_FEAT_SIZE = 0

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


def standarize(df):
    scaler = StandardScaler()
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

def train_one_epoch(model, train_loader, optimizer, loss_function, epoch):
        model.train(True)
        # print(f'Epoch: {epoch + 1}')
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
                # print('Batch {0}, Loss: {1:.3f}'.format(batch_index+0, avg_loss_across_batches))
                running_loss = 0.0
        # print()

def validate_one_epoch(model, test_loader, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        X_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    # print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    # print('***************************************************')
    # print()

def train_model(batch_size, num_epochs, learning_rate, hidden_size, num_stacked_layers, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM(1, hidden_size, num_stacked_layers)
    model.to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, loss_function, epoch)
        validate_one_epoch(model, test_loader, loss_function)

    with torch.no_grad():
        predicted = model(test_dataset.X.to(device)).to('cpu').numpy()

    mape = mean_absolute_percentage_error(test_dataset.y, predicted)

    return predicted, mape

def train_model_kfold(k, model_params, traindata, batch_size = 16, num_epochs = 10):
    X, y = traindata.X, traindata.y
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_mape_scores = []
    best_mape = float('inf')
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        # test_dataset = TimeSeriesDataset(testdata)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = LSTM(input_size=1, hidden_size=model_params.get("hidden_size"), num_stacked_layers=model_params.get("num_stacked_layers"))
        model.to(device)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("learning_rate"))

        # Train and validate for each epoch
        for epoch in range(num_epochs):
            train_one_epoch(model, train_loader, optimizer, loss_function, epoch)

        # Validate
        with torch.no_grad():
            predictions = []
            ground_truths = []
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                predictions.append(output.cpu().numpy())
                ground_truths.append(y_batch.cpu().numpy())

            # Flatten predictions and ground truths
            predictions = np.concatenate(predictions)
            ground_truths = np.concatenate(ground_truths)

            # Calculate MAPE for the current fold
            fold_mape = mean_absolute_percentage_error(ground_truths, predictions)
            print(f"Fold {fold + 1} MAPE: {fold_mape}")
            fold_mape_scores.append(fold_mape)

            # predicted = best_model(test_dataset.X.to(device)).to('cpu').numpy()
            # mape_best_model = mean_absolute_percentage_error(test_dataset.y, predicted)

            # update best model if MAPE is lower
            if fold_mape < best_mape:
                best_mape = fold_mape
                best_model = dc(model)

    # Calculate the average MAPE across all folds
    avg_mape = np.mean(fold_mape_scores)
    print(f"Average MAPE across {k} folds: {avg_mape}")

    return best_model, avg_mape, fold_mape_scores

def main(args):
    """ Main entry point of the app """
    training_file = args.training_file
    testing_file = args.testing_file
    # HERE GOES YOUR CODE TO CALCULATE THE MAPE
    # FEEL FREE TO IMPLEMENT HELPER FUNCTIONS

    train_data, test_data = read_files(training_file, testing_file)
    # print(train_data) # 5 columns, 1629 rows
    # print(test_data) # 5 columns, 23 rows

    lookback = 7
    # # Adds lookback columns to the dataframe (the values of last close from the previous day up until "lookback" days backwards)
    shifted_training_df = prepare_dataframe_for_lstm(train_data, lookback)
    shifted_testing_df = prepare_dataframe_for_lstm(test_data, lookback)
    FEATURE_AMOUNT = shifted_training_df.shape[1] # One less "feature" than og df because date is now the index of the row, no longer a feature
    columns = shifted_training_df.columns.tolist() # Create a list of column names to be able to split data later on
    target_feature_index = columns.index("Last Close")
    # print(shifted_training_df) # (FEATURE_AMOUNT) columns, 1629 - lookback rows
    # print(shifted_testing_df) # (FEATURE_AMOUNT) columns, 23 - lookback rows

    shifted_training_df_as_np = shifted_training_df.to_numpy() # Turn into numpy 2D array
    shifted_testing_df_as_np = shifted_testing_df.to_numpy() # Turn into numpy 2D array

    shifted_training_df_as_np = standarize(shifted_training_df_as_np) # Flatten data
    shifted_testing_df_as_np = standarize(shifted_testing_df_as_np) # Flatten data

    # Take all columns except for the actual last close column as X (so open, high, low, and the lookback values of last close)
    X_train = np.delete(shifted_training_df_as_np, target_feature_index, axis=1)
    X_test = np.delete(shifted_testing_df_as_np, target_feature_index, axis=1)
    # # Take the actual last close column as the feature to predict
    y_train= shifted_training_df_as_np[:, target_feature_index]
    y_test = shifted_testing_df_as_np[:, target_feature_index]
    # Flip the training feature dataframe so that the oldest of the lookback values is at the front of the dataframe so the LSTM model can learn the trend going forwards in time
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

    train_data = TimeSeriesDataset(X_train, y_train)
    test_data = TimeSeriesDataset(X_test, y_test)

    # Stel je hebt de mapes-lijst zoals in jouw code
    # mapes = []
    # for i in range(20):
    #     learning_rate = 0.001 + 0.01 * i
    #     predicted, mape = train_model(
    #         batch_size=16,
    #         num_epochs=25,
    #         learning_rate=learning_rate,
    #         hidden_size=10,
    #         num_stacked_layers=3,
    #         train_dataset=train_data,
    #         test_dataset=test_data)
    #     mapes.append([learning_rate, mape])
    #     print(f'Iteration {i+1} done')
    #     print(f'Learing rate: {learning_rate}')
    #     print(f'MAPE: {mape}')
    #     print('***************************************************')
    #
    # # Extract de learning rates en MAPE-waarden
    # learning_rates = [x[0] for x in mapes]
    # mape_values = [x[1] for x in mapes]
    #
    # # Plot de MAPE-waarden tegen de learning rates
    # plt.plot(learning_rates, mape_values, marker='o')
    # plt.xlabel('Learning Rate')
    # plt.ylabel('MAPE')
    # plt.title('MAPE in functie van de Learning Rate')
    # plt.show()

    model_params = {
        "hidden_size": 10,
        "num_stacked_layers": 3,
        "learning_rate": 0.001
    }

    best_model, avg_mape, fold_mape_scores = train_model_kfold(
        k=5,
        model_params=model_params,
        traindata=train_data,
        batch_size=16,
        num_epochs=10
    )

    with torch.no_grad():
        predicted = best_model(test_data.X.to(device)).to('cpu').numpy()

    mape_best_model = mean_absolute_percentage_error(test_data.y, predicted)

    print(mape_best_model)

    # plt.plot(test_data.y, label='actual close')
    # plt.plot(predicted, label='predicted close')
    # plt.xlabel('Day')
    # plt.ylabel('close')
    # plt.legend()
    # plt.show()





    # # plt.plot(np.l, label = 'Actual Last Close')
    # # plt.plot(predicted, label = 'Predicted Last Close')
    # # plt.xlabel('Day')
    # # plt.ylabel('Close')
    # # plt.legend()
    # # plt.show()
    #
    # # plt.plot(train_data['Date'], train_data['Last Close'])
    # # plt.show()
    # # train_values = train_data['Last Close'].values.reshape(-1, 1)
    # # test_values = test_data['Last Close'].values.reshape(-1, 1)
    #
    # # plot_data(train_data)
    #
    # # train_values = normalize(train_values)
    #
    # # seq_length = 2
    # # x, y = create_sequences(train_values, seq_length)



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