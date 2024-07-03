import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import seaborn as sns
from tqdm import tqdm
FORECAST_HORIZON = 1
N_TRAIN_MONTHS = 80
N_VALIDATION_MONTHS = 30 
N_TEST_MONTHS = 10

def create_split_df(df, title, forecast_horizon):
    X = df[title]
    y = df[title].shift(-forecast_horizon)
    X = X[:-forecast_horizon]
    y = y[:-forecast_horizon]
    data = pd.DataFrame(index=df.index[:-forecast_horizon])
    data = pd.concat([X, y], axis=1)
    data.columns= ['X_title', 'y_title']
    #approximate the data to the 4 decimal
    data = data.round(4)
    data.to_csv('data/{}_train_data.csv'.format(title))
    return data

def get_train_valid_test_df(df, n_train_months, n_valid_months, n_test_months):
    X = df.drop(columns='y_title')
    y = df['y_title']
    N_TOTAL_MONTHS = n_train_months + n_valid_months + n_test_months
    days_in_month = len(df) // N_TOTAL_MONTHS

    train_df = df.iloc[:n_train_months*days_in_month]
    valid_df = df.iloc[n_train_months*days_in_month:n_train_months*days_in_month+n_valid_months*days_in_month]
    test_df = df.iloc[n_train_months*days_in_month+n_valid_months*days_in_month:]

    return train_df, valid_df, test_df

def get_best_model(train_df, valid_df, title):
    #create a file with the best configuration in the folder "config/" for the title passed
    if os.path.exists('config/best_params_{}.json'.format(title)):
        with open('config/best_params_{}.json'.format(title), 'r') as f:
            print("reading best configuration for {}".format(title))
            best_params = json.load(f)
        return SVR(**best_params)
    
    X_train = train_df.drop(columns='y_title')
    y_train = train_df['y_title']
    X_valid = valid_df.drop(columns='y_title')
    y_valid = valid_df['y_title']

    models = []
    for C in [0.005, 0.01, 0.1, 1, 10, 100]:
        for gamma in ['scale', 'auto']:
            for kernel in ['rbf', 'linear']:
                for epsilon in [0.01, 0.1, 0.5, 1]:
                    models.append(SVR(C=C, gamma=gamma, kernel=kernel, epsilon=epsilon))
    #train all the models and print the mse, keep the best one
    lowest_mse = float('inf')
    with tqdm(total=len(models), desc="finding best SVR model for {}".format(title), ncols=100) as pbar:
        for model in models:
            model.fit(X_train.values, y_train)
            y_pred = model.predict(X_valid.values)
            mse = mean_squared_error(y_valid, y_pred)
            if mse < lowest_mse:
                best_model = model
                lowest_mse = mse
            pbar.update(1)

    #dump the best params on file
    os.makedirs('config', exist_ok=True)
    with open('config/best_params_{}.json'.format(title), 'w') as f:
        json.dump(best_model.get_params(), f)

    print("best model found, best params: {}".format(best_model.get_params()))
    return best_model

def scale_data(X_train, X_valid, X_test):
    #scale the data but keep the indexes
    print("scaling data")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_valid = pd.DataFrame(scaler.transform(X_valid), index=X_valid.index, columns=X_valid.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train, X_valid, X_test

def plot_details_pred(y_test, y_pred, title):
    COLORS = sns.color_palette("hls", 4)
    fig, ax = plt.subplots(1)
    ax = sns.lineplot(data=y_test, color=COLORS[3], label='Attuale')
    ax.plot(y_test.index, y_pred, c=COLORS[1], label='Predizione')
    ax.set(title="{} stock price DETAIL OF - actual vs. predicted".format(title), xlabel='Date', ylabel='Price ($)')
    ax.legend()
    fig.set_size_inches(12, 8)
    if not os.path.exists('images/predictions'):
        os.makedirs('images/predictions', exist_ok=True)
    plt.savefig('images/predictions/{}_DETAIL_predictions.png'.format(title))
    # plt.show()

def plot_predictions(y_test, y_pred, y_train, title):
    COLORS = sns.color_palette("hls", 4)
    fig, ax = plt.subplots(1)
    ax = sns.lineplot(data=y_train, color=COLORS[0], label='Attuale')
    ax = sns.lineplot(data=y_test, color=COLORS[3], label='Test')
    ax.plot(y_test.index, y_pred, c=COLORS[1], label='Predizione')

    ax.set(title="{} stock price - actual vs. predicted".format(title), xlabel='Date', ylabel='Price ($)')
    ax.legend()
    fig.set_size_inches(12, 8)
    if not os.path.exists('images/predictions'):
        os.makedirs('images/predictions', exist_ok=True)
    plt.savefig('images/predictions/{}_predictions.png'.format(title))
    #plt.show()

data_df = pd.read_csv('data/data.csv', index_col=0, parse_dates=True)

# title_df = create_split_df(data_df, 'NVDA', FORECAST_HORIZON)

# train_df, valid_df, test_df = get_train_valid_test_df(title_df, N_TRAIN_MONTHS, N_VALIDATION_MONTHS, N_TEST_MONTHS)

# X_train, X_valid, X_test = scale_data(train_df.drop(columns='y_title'), valid_df.drop(columns='y_title'), test_df.drop(columns='y_title'))
# train_df = train_df.copy()
# valid_df = valid_df.copy()
# test_df = test_df.copy()

# train_df['X_title'] = X_train
# valid_df['X_title'] = X_valid
# test_df['X_title'] = X_test

# best_model = get_best_model(train_df, valid_df, 'NVDA')

# #refit the best model on the training_set + validation_set
# X_train = pd.concat([train_df.drop(columns='y_title'), valid_df.drop(columns='y_title')])
# y_train = pd.concat([train_df['y_title'], valid_df['y_title']])
# best_model.fit(X_train.values, y_train)

# X_test = test_df.drop(columns='y_title')
# y_test = test_df['y_title']

# predictions = []
# with tqdm(total=len(X_test), desc="predicting recursively on test set", ncols=100) as pbar:
#     for row in X_test.iterrows():
#         predictions.append(best_model.predict(row[1].values.reshape(1, -1)))
#         #update the training set with the new prediction
#         X_train = X_train._append(row[1])
#         y_add_series = pd.Series(predictions[-1], index=[row[0]])
#         y_train = y_train._append(y_add_series)
#         best_model.fit(X_train.values, y_train)
#         pbar.update(1)


# #print error metrics
# print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))
# print("Root mean squared error: {}".format(np.sqrt(mean_squared_error(y_test, predictions))))
# print("Mean absolute error: {}".format(mean_absolute_error( y_test, predictions)))
# print("R2 score: {}".format(best_model.score(X_test.values, y_test)))

# plot_predictions(y_test, predictions, y_train, 'NVDA')
# plot_details_pred(y_test, predictions, 'NVDA')
# plt.show()

tickers = ["NVDA", "INTC", "HII", "TDG", "JPM", "BAC"]

for ticker in tickers:
    if not os.path.exists('data/{}_predictions.csv'.format(ticker)):
        title_df = create_split_df(data_df, ticker, FORECAST_HORIZON)
        train_df, valid_df, test_df = get_train_valid_test_df(title_df, N_TRAIN_MONTHS, N_VALIDATION_MONTHS, N_TEST_MONTHS)
        X_train, X_valid, X_test = scale_data(train_df.drop(columns='y_title'), valid_df.drop(columns='y_title'), test_df.drop(columns='y_title'))
        train_df = train_df.copy()
        valid_df = valid_df.copy()
        test_df = test_df.copy()

        train_df['X_title'] = X_train
        valid_df['X_title'] = X_valid
        test_df['X_title'] = X_test

        best_model = get_best_model(train_df, valid_df, ticker)

        #refit the best model on the training_set + validation_set
        X_train = pd.concat([train_df.drop(columns='y_title'), valid_df.drop(columns='y_title')])
        y_train = pd.concat([train_df['y_title'], valid_df['y_title']])
        best_model.fit(X_train.values, y_train)

        X_test = test_df.drop(columns='y_title')
        y_test = test_df['y_title']

        predictions = []
        with tqdm(total=len(X_test), desc="predicting recursively on test set", ncols=100) as pbar:
            for row in X_test.iterrows():
                predictions.append(best_model.predict(row[1].values.reshape(1, -1)))
                #update the training set with the new prediction
                X_train = X_train._append(row[1])
                y_add_series = pd.Series(predictions[-1], index=[row[0]])
                y_train = y_train._append(y_add_series)
                best_model.fit(X_train.values, y_train)
                pbar.update(1)

        save_predictions = pd.DataFrame({'real_y': y_test, 'predicted_y': predictions})
        save_predictions.to_csv('data/{}_predictions.csv'.format(ticker))
    #print error metrics
    y_test = pd.read_csv('data/{}_predictions.csv'.format(ticker), index_col=0, parse_dates=True)['real_y']
    predictions = pd.read_csv('data/{}_predictions.csv'.format(ticker), index_col=0, parse_dates=True)['predicted_y']
    print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))
    print("Root mean squared error: {}".format(np.sqrt(mean_squared_error(y_test, predictions))))
    print("Mean absolute error: {}".format(mean_absolute_error( y_test, predictions)))
    print("R2 score: {}".format(best_model.score(X_test.values, y_test)))

    
    plot_predictions(y_test, predictions, y_train, ticker)
    plot_details_pred(y_test, predictions, ticker)

    print("done with {}".format(ticker))