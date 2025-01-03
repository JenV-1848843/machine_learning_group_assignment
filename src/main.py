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
from sklearn.model_selection import KFold, TimeSeriesSplit

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
    

class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        # Ensure no division by zero
        epsilon = 1e-8  # Small value to prevent division by zero
        return torch.mean(torch.abs((targets - predictions) / (targets + epsilon)) * 100)
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path="best_model.pth"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            with open(self.path, 'wb') as f:
                torch.save(model.state_dict(), f)  # Save the best model

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        # The final layer should be of size one as we're predicting one output value (close stock value)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        # _ here is the updated (h0, c0) tuple which we don't care about in this case
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out[:, -1, :])
        # transform the output from the fc layer to the appropriate size
        out = self.fc(out)
        return out

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['ShortEMA'] = df['Last Close'].ewm(span=short_window, adjust=False).mean()
    df['LongEMA'] = df['Last Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['ShortEMA'] - df['LongEMA']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

def calculate_moving_average(df, window=20):
    df[f'MA_{window}'] = df['Last Close'].rolling(window=window).mean()

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    rolling_mean = df['Last Close'].rolling(window=window).mean()
    rolling_std = df['Last Close'].rolling(window=window).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std_dev)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std_dev)

def calculate_rsi(df, window=14):
    delta = df['Last Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    # Calculate technical indicators
    # calculate_macd(df)
    # calculate_moving_average(df, window=20)
    # calculate_bollinger_bands(df, window=20)
    # calculate_rsi(df, window=14)

    for i in range(1, n_steps + 1):
        df[f'Last Close(t-{i})'] = df['Last Close'].shift(i)

    df.drop(columns=['Open'], inplace=True)

    # Drop rows with NaN values due to feature calculations or lagging
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
    model.eval()
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        X_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    return avg_loss_across_batches

    # print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    # print('***************************************************')
    # print()


def train_model_kfold(k, model_params, traindata, batch_size=16, num_epochs=100):
    X, y = traindata.X, traindata.y
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gebruik TimeSeriesSplit voor tijdreeksen
    kf = TimeSeriesSplit(n_splits=k)

    fold_mape_scores = []
    best_mape = float('inf')
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{k}")

        # Train en validatie splits
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Datasets en DataLoaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model initialisatie
        model = LSTM(
            input_size=1,
            hidden_size=model_params.get("hidden_size"),
            num_stacked_layers=model_params.get("num_stacked_layers"),
            dropout_rate=model_params.get("dropout_rate")
        )
        model.to(device)

        loss_function = MAPELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("learning_rate"))

        # Early Stopping initialisatie
        model_path = f"fold_{fold + 1}.pth"
        early_stopping = EarlyStopping(patience=model_params.get("patience"), path=model_path)

        # Train en validatie loop
        for epoch in range(num_epochs):
            train_one_epoch(model, train_loader, optimizer, loss_function, epoch)

            val_loss = validate_one_epoch(model, val_loader, loss_function)
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Laad het beste model voor deze fold
        model.load_state_dict(torch.load(model_path))

        # Validatie op de validatieset
        with torch.no_grad():
            val_predictions = []
            val_ground_truths = []
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_predictions.append(output.cpu().numpy())
                val_ground_truths.append(y_batch.cpu().numpy())

            val_predictions = np.concatenate(val_predictions)
            val_ground_truths = np.concatenate(val_ground_truths)

            fold_mape = mean_absolute_percentage_error(val_ground_truths, val_predictions)
            print(f"Fold {fold + 1} Validation MAPE: {fold_mape:.4f}")
            fold_mape_scores.append(fold_mape)

        # Update het beste model
        if fold_mape < best_mape:
            best_mape = fold_mape
            best_model = dc(model)

    # Gemiddelde scores berekenen
    avg_val_mape = np.mean(fold_mape_scores)
    # avg_test_mape = np.mean(fold_test_mape_scores)

    print(f"\nAverage Validation MAPE across {k} folds: {avg_val_mape:.4f}")
    # print(f"Average Test MAPE across {k} folds: {avg_test_mape:.4f}")

    return best_model

def parameter_tuning(parameter_names, parameters_values, train_data, testdata, k=5, num_epochs=50, batch_size=16):
    mapes = []  # Store the MAPE results for each configuration

    # Generate combinations of parameters to test
    from itertools import product
    all_param_combinations = list(product(*parameters_values))

    lowest_mape = float('inf')
    best_combination = None

    for param_comb in all_param_combinations:
        model_params = dict(zip(parameter_names, param_comb))

        # Train the model with the current parameter combination
        best_model = train_model_kfold(k, model_params, train_data, batch_size, num_epochs)
        
        # Predict on the test data
        with torch.no_grad():
            predicted = best_model(testdata.X.to(device)).to('cpu').numpy()
        
        # Calculate the MAPE on the test set
        mape = mean_absolute_percentage_error(testdata.y, predicted)
        mapes.append((param_comb, mape))

        # Print the results for this parameter combination
        print(f'Parameter Combination: {param_comb}')
        print(f'MAPE on test dataset: {mape}')
        print('***************************************************')

        # Update the best combination if the current MAPE is lower
        if mape < lowest_mape:
            lowest_mape = mape
            best_combination = param_comb

    print(f'Lowest MAPE: {lowest_mape}')
    print(f'Best parameter combination: {best_combination}')

    # Extract parameter combinations and corresponding MAPE values
    param_combinations = [x[0] for x in mapes]
    mape_values = [x[1] for x in mapes]

    # Plot the MAPE values against the parameter combinations
    for i, param_name in enumerate(parameter_names):
        plt.figure(figsize=(10, 6))
        param_values = [comb[i] for comb in param_combinations]
        plt.plot(param_values, mape_values, marker='o')
        plt.xlabel(param_name)
        plt.ylabel('MAPE')
        plt.title(f'MAPE vs {param_name}')
        plt.show()


def run_one_model(train_data, test_data, model_params, num_epochs=50, batch_size=16, multiple_models=False):
    best_model = train_model_kfold(
        k=5,
        model_params=model_params,
        traindata=train_data,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    with torch.no_grad():
        predicted = best_model(test_data.X.to(device)).to('cpu').numpy()

    mape_best_model = mean_absolute_percentage_error(test_data.y, predicted)
    print("MAPE of model: ", mape_best_model)

    if not multiple_models:
        plot_predictions(test_data.y, predicted, title=f'Model Predictions (MAPE: {mape_best_model:.4f})')

    return mape_best_model, predicted

def run_multiple_models(train_data, test_data, model_params, n, num_epochs=50, batch_size=16):
    mapes = {}
    predictions = {}


    for i in range(n):
        print(f"============ Model {i+1}/{n} ============")
        mape, predicted = run_one_model(train_data, test_data, model_params, num_epochs, batch_size, multiple_models=True)
        mapes[f"Model {i+1}"] = mape
        predictions[f"Model {i+1}"] = predicted

    average = 0
    best_mape = float('inf')
    worst_mape = float('-inf')
    best_model = None
    worst_model = None

    for model, mape in mapes.items():
        average += mape
        if mape < best_mape:
            best_mape = mape
            best_model = model
        if mape > worst_mape:
            worst_mape = mape
            worst_model = model

    average /= n

    print(f'Average MAPE: {average:.4f}')
    print(f'Best MAPE: {best_mape:.4f}')
    print(f'Worst MAPE: {worst_mape:.4f}')

    plot_average_predictions(list(mapes.keys()), list(mapes.values()), average)
    plot_predictions(test_data.y, predictions[best_model], title=f'Best Model Predictions: {best_model} (MAPE: {best_mape:.4f})')
    plot_predictions(test_data.y, predictions[worst_model], title=f'Worst Model Predictions: {worst_model} (MAPE: {worst_mape:.4f})')

    return mapes, predictions


def plot_predictions(y_true, y_pred, title='Predictions'):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Last Close')
    plt.plot(y_pred, label='Predicted Last Close')
    plt.xlabel('Day')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_average_predictions(x_models, y_MAPE, average_MAPE):
    plt.figure(figsize=(10, 6))
    plt.plot(y_MAPE, marker='o')
    plt.axhline(y=average_MAPE, color='r', linestyle='--', label='Average MAPE')
    plt.xticks(ticks=range(len(x_models)), labels=x_models, rotation='vertical', fontsize=6)
    plt.xlabel('Model')
    plt.ylabel('MAPE')
    plt.title('MAPE vs Model')
    plt.legend()
    plt.show()

def main(args):
    """ Main entry point of the app """
    training_file = args.training_file
    testing_file = args.testing_file
    # HERE GOES YOUR CODE TO CALCULATE THE MAPE
    # FEEL FREE TO IMPLEMENT HELPER FUNCTIONS

    train_data, test_data = read_files(training_file, testing_file)
    # print(train_data) # 5 columns, 1629 rows
    # print(test_data) # 5 columns, 23 rows

    lookback = 3
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

    # vvvvvvvvvvvvv PARAMETER TUNING vvvvvvvvvvvvv
    # parameter_names = ["hidden_size", "num_stacked_layers", "learning_rate", "dropout_rate", "patience"]
    # parameters_values = [
    #     [110, 120, 130],         # hidden_size values
    #     [2, 3, 4],              # num_stacked_layers values
    #     [0.00005, 0.0001],     # learning_rate values
    #     [0.1],         # dropout_rate values
    #     [3, 4, 5]              # patience
    # ]

    # parameter_tuning(parameter_names, parameters_values, train_data, test_data)

    # ======== END OF PARAMETER TUNING =========

    # vvvvvvvvvvvvv ONE MODEL TRAINING vvvvvvvvvvvvv

    # # Best model parameters:
    # hidden_size = 110
    # num_stacked_layers = 4
    # learning_rate = 0.0001
    # dropout_rate = 0.1
    # patience = 4

    # model_params = {
    #     "hidden_size": 110,
    #     "num_stacked_layers": 4,
    #     "learning_rate": 0.0001,
    #     "dropout_rate": 0.1,
    #     "patience": 4
    # }

    # mape_best_model, predictions = run_one_model(train_data, test_data, model_params)

    # ======== END OF ONE MODEL TRAINING =========

    # vvvvvvvvvvvvv MULTIPLE MODELS TRAINING vvvvvvvvvvvvv

    # # Best model parameters:
    # hidden_size = 110
    # num_stacked_layers = 4
    # learning_rate = 0.0001
    # dropout_rate = 0.1
    # patience = 4
    
    # n = 5
    # model_params = {
    #     "hidden_size": 110,
    #     "num_stacked_layers": 4,
    #     "learning_rate": 0.0001,
    #     "dropout_rate": 0.1,
    #     "patience": 4
    # }

    # mapes, predictions = run_multiple_models(train_data, test_data, model_params, n)

    # ======== END OF MULTIPLE MODELS TRAINING =========


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