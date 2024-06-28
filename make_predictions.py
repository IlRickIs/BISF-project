import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import seaborn as sns
#prepariamo il dataset indipendente e dipendente
def create_split_df(df, title, forecast_horizon):
    X = df[title]
    y = df[title].shift(-forecast_horizon)
    X = X[:-forecast_horizon]
    y = y[:-forecast_horizon]
    data = pd.DataFrame(index=df.index[:-forecast_horizon])
    data = pd.concat([X, y], axis=1)
    data.columns= ['X_title', 'y_title']
    return data

def make_train_test_split(df, n_train_months):
    X = df.drop(columns='y_title')
    y = df['y_title']
    X_train = X.iloc[:n_train_months*21]
    y_train = y.iloc[:n_train_months*21]
    X_test = X.iloc[n_train_months*21-1:]
    y_test = y.iloc[n_train_months*21-1:]
    return X_train, X_test, y_train, y_test

def get_best_svr(X_train, y_train, title):
    #create a file with the best configuration in the folder "config/" for the title passed
    if os.path.exists('config/best_params_{}.json'.format(title)):
        with open('config/best_params_{}.json'.format(title), 'r') as f:
            print("reading best configuration for {}".format(title))
            best_params = json.load(f)
        return SVR(**best_params)
    print("searching best configuration for {}".format(title))
    param_grid = {'C': [0.1, 1, 10, 100], 
                  'gamma': [1, 0.5, 0.2, 0.1, 0.01], 
                  'kernel': ['rbf', 'linear', 'poly'],
                  'epsilon': [0.01, 0.1, 0.5, 1]}
    grid = GridSearchCV(SVR(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    #dump the best params on file
    os.makedirs('config', exist_ok=True)
    with open('config/best_params_{}.json'.format(title), 'w') as f:
        json.dump(grid.best_params_, f)
    return grid.best_estimator_


def scale_data(X_train, X_test):
    #scale the data but keep the indexes
    print("scaling data")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train, X_test

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
    plt.show()

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
    plt.show()

FORECAST_HORIZON = 10
N_TRAIN_MONTHS = 90
N_TEST_MONTHS = 30 

data_df = pd.read_csv('data/data.csv', index_col=0, parse_dates=True)
# nvdia_df = create_split_df(data_df, 'NVDA', FORECAST_HORIZON)

# X_train, X_test, y_train, y_test = train_test_split(nvdia_df.drop(columns= 'X_title'), 
#                                                     nvdia_df['y_title'], test_size=N_TEST_MONTHS/120)
# X_train, X_test, y_train, y_test = make_train_test_split(nvdia_df, N_TRAIN_MONTHS, N_TEST_MONTHS)
# print(type(X_train))

# X_train, X_test = scale_data(X_train, X_test)

# svr = get_best_svr(X_train, y_train, 'NVDA')
# svr.fit(X_train, y_train)
# X_test = X_test.sort_index()
# y_pred = svr.predict(X_test)
# predictions_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
# predictions_df.to_csv('data/NVDA_predictions.csv')
# #reorder y_test and y_pred by date
# y_test = y_test.sort_index()
# y_train = y_train.sort_index()
# print(y_pred.shape)
# print('MSE:', mean_squared_error(y_test, y_pred))
# print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred)))
# print('MAE:', mean_absolute_error(y_test, y_pred))

# plot_predictions(y_test, y_pred, y_train, 'NVDA')

#make prediciton for all the tiles in data_df
for title in data_df.columns:
    print("building a model to predict the price for {}".format(title))
    title_df = create_split_df(data_df, title, FORECAST_HORIZON)
    X_train, X_test, y_train, y_test = make_train_test_split(title_df, N_TRAIN_MONTHS)
    X_train, X_test = scale_data(X_train, X_test)
    svr = get_best_svr(X_train, y_train, title)
    svr.fit(X_train, y_train)
    X_test = X_test.sort_index()
    y_pred = svr.predict(X_test)
    y_test = y_test.sort_index()
    predictions_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
    predictions_df.to_csv('data/{}_predictions.csv'.format(title))
    print('{} MSE:'.format(title), mean_squared_error(y_test, y_pred))
    print('{} RMSE:'.format(title),np.sqrt(mean_squared_error(y_test, y_pred)))
    print('{} MAE:'.format(title), mean_absolute_error(y_test, y_pred))
    plot_predictions(y_test, y_pred, y_train, title)
    plot_details_pred(y_test, y_pred, title)