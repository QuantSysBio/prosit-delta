
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from deltapro.constants import BLOSUM6_1_VALUES

MAX_DEPTH = 17

FEATURE_SET = [
    'spectralAngle',
    'blosumC',
    'blosumN',
    'massDiff',
    'cNeighbourBlosum',
    'nNeighbourBlosum',
    'relPos',
    'bIntesAtC',
    'yIntesAtN',
    'yIntesAtLoc',
    'bIntesAtLoc',
    'yMatchedIntesAtN',
    'bMatchedIntesAtC',
    'yMatchedIntesAtLoc',
    'bMatchedIntesAtLoc',
    'cNeighbourOxidation',
    'nNeighbourOxidation',
    'cOxidation',
    'nOxidation',
]
TARGET_VARIABLE = 'specAngleDiff'

def run_training(train_df, max_d):
    reg = RandomForestRegressor(
        max_depth=max_d,
    )
    reg.fit(train_df[FEATURE_SET].values, train_df[TARGET_VARIABLE])
    return reg

def assess_model(model, dataset, title, plot):
    dataset['predictedDiff'] = model.predict(dataset[FEATURE_SET].values)

    mae_ = mean_absolute_error(dataset[TARGET_VARIABLE], dataset['predictedDiff'])

    if plot:
        dataset.plot.scatter(TARGET_VARIABLE, 'predictedDiff')
        plt.show()
    print(f'{title} R2score: {r2_score(dataset[TARGET_VARIABLE], dataset["predictedDiff"])}')
    print(f'{title} Pearson R: {pearsonr(dataset[TARGET_VARIABLE], dataset["predictedDiff"])[0]}')
    print(f'{title} MAE: {mae_}')
    return mae_

def show_importances(model):
    plt.barh(FEATURE_SET, model.feature_importances_)
    plt.show()

def save_model(model, folder):
    file_name = f'{folder}/reg.pkl'
    pickle.dump(model, open(file_name, 'wb'))

def edit_features(working_df):
    if 'intesAtC' in FEATURE_SET:
        working_df['intesAtC'] = working_df['bIntesAtC'] + working_df['yIntesAtC']
        working_df['intesAtN'] = working_df['bIntesAtN'] + working_df['yIntesAtN']
        working_df['intesAtLoc'] = working_df['bIntesAtLoc'] + working_df['yIntesAtLoc']
    if 'matchedIntesAtC' in FEATURE_SET:
        working_df['matchedIntesAtC'] = working_df['bMatchedIntesAtC'] + working_df['yMatchedIntesAtC']
        working_df['matchedIntesAtN'] = working_df['bMatchedIntesAtN'] + working_df['yMatchedIntesAtN']
        working_df['matchedIntesAtLoc'] = working_df['bMatchedIntesAtLoc'] + working_df['yMatchedIntesAtLoc']
    if 'errsAtLoc' in FEATURE_SET:
        working_df['errsAtC'] = working_df['yErrsAtC'] + working_df['bErrsAtC']
        working_df['errsAtN'] = working_df['yErrsAtN'] + working_df['bErrsAtN']
        working_df['errsAtLoc'] = working_df['yErrsAtLoc'] + working_df['yErrsAtLoc']
    if 'cNeighbourBlosum' in FEATURE_SET:
        working_df['cNeighbourBlosum'] = working_df['cNeighbour'].apply(lambda x : BLOSUM6_1_VALUES.get(x, 0.0))
        working_df['nNeighbourBlosum'] = working_df['nNeighbour'].apply(lambda x : BLOSUM6_1_VALUES.get(x, 0.0))
    return working_df

def train_model(folder):
    train_df = pd.read_csv(f'{folder}/trainData.csv')
    train_df = train_df.fillna(0)
    test_df = pd.read_csv(f'{folder}/testData.csv')

    test_df = test_df.fillna(0)

    train_df = edit_features(train_df)
    test_df = edit_features(test_df)

    train_maes = []
    test_maes = []

    do_plot=False
    do_plot=True
    print(f'\nFor Max Depth {MAX_DEPTH}:\n')
    model = run_training(train_df, MAX_DEPTH)
    train_mae = assess_model(model, train_df, 'Train', plot=do_plot)
    print()
    test_mae = assess_model(model, test_df, 'Test', plot=do_plot)
    train_maes.append(train_mae)
    test_maes.append(test_mae)
    save_model(model, folder)
    show_importances(model)

