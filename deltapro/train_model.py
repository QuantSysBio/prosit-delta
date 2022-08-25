
import os
import pickle
from tkinter.font import families

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error, r2_score
import sklearn
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from deltapro.constants import BLOSUM6_1_VALUES


FEATURE_SET = [
    'spectralAngle',
    'blosumC',
    'blosumN',
    'nTermDist',
    'cTermDist',
    'charge',
    'cNeighbourBlosum',
    'nNeighbourBlosum',
    'yErrsAtLoc',
    'bErrsAtLoc',
    'collisionEnergy',
    'bIntesAtC',
    'yIntesAtN',
    'yIntesAtLoc',
    'bIntesAtLoc',
    'yMatchedIntesAtN',
    'bMatchedIntesAtC',
    'yMatchedIntesAtLoc',
    'bMatchedIntesAtLoc',
    'cOxidation',
    'nOxidation',
    'flipYNewIntensity',
    'flipBNewIntensity',
    'matchedCoverage',
]
TARGET_VARIABLE = 'specAngleDiff'

METHOD = 'xgb'
PARAM_SETS = {
    "learning_rate"    : [0.15, 0.20, 0.30 ] ,
    "max_depth"        : [ 14, 15, 16, 17, 18],
    "min_child_weight" : [ 1, 2],
    "gamma"            : [ 0.0, 0.1, 0.2 ],
    "colsample_bytree" : [ 0.7, 0.9 ],
}


def run_training(train_df, entry, optimised_settings):
    reg = xgb.XGBRegressor(n_jobs=25, **optimised_settings[entry])
    
    reg.fit(train_df[FEATURE_SET].values, train_df[TARGET_VARIABLE])
    return reg

def assess_model(model, dataset, title, identifier):
    dataset[f'predictedDiff{identifier}'] = model.predict(dataset[FEATURE_SET].values)

    model_stats = {
        f'{title}mae': median_absolute_error(dataset[TARGET_VARIABLE], dataset[f'predictedDiff{identifier}']),
        f'{title}r2': r2_score(dataset[TARGET_VARIABLE], dataset[f'predictedDiff{identifier}']),
        f'{title}pearson': pearsonr(dataset[TARGET_VARIABLE], dataset[f'predictedDiff{identifier}'])[0],
        f'{title}spearmanr': spearmanr(dataset[TARGET_VARIABLE], dataset[f'predictedDiff{identifier}'])[0],
    }
    return dataset, model_stats

def save_importances(model, output_folder, identifier):
    importance_df = pd.DataFrame({
        'feature': pd.Series(FEATURE_SET),
        'importances': pd.Series(model.feature_importances_)
    })
    importance_df.to_csv(
        f'{output_folder}/importances/model{identifier}.csv',
        index=False,
    )

def save_model(model, folder, identifier):
    file_name = f'{folder}/model/reg{identifier}.pkl'
    pickle.dump(model, open(file_name, 'wb'))
    model_size = os.path.getsize(file_name)
    return model_size

def edit_features(working_df):
    # working_df['flip'] = working_df.apply(
    #     lambda x : (x['peptide'][:int(x['flipInd']) - 1] +
    #         x['peptide'][int(x['flipInd'])] +
    #         x['peptide'][int(x['flipInd']-1)] +
    #         x['peptide'][int(x['flipInd'])+1:]),
    #     axis=1
    # )

    working_df['pepLen'] = working_df['peptide'].apply(len)
    working_df['nTermDist'] = working_df['flipInd'].apply(int)
    working_df['cTermDist'] = working_df['pepLen'] - working_df['nTermDist']

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
        working_df['errsAtLoc'] = working_df['yErrsAtLoc'] + working_df['bErrsAtLoc']
    if 'cNeighbourBlosum' in FEATURE_SET:
        working_df['cNeighbourBlosum'] = working_df['cNeighbour'].apply(lambda x : BLOSUM6_1_VALUES.get(x,-5.0))
        working_df['nNeighbourBlosum'] = working_df['nNeighbour'].apply(lambda x : BLOSUM6_1_VALUES.get(x, -5.0))
    return working_df


def hpt_job(train_df):
    reg = xgb.XGBRegressor()
    cv = RandomizedSearchCV(
        reg, param_distributions=PARAM_SETS, n_iter=50, scoring='r2', n_jobs=15, cv=5, random_state=42, verbose=3,
    )
    searched_cv = cv.fit(train_df[FEATURE_SET].values, train_df[TARGET_VARIABLE].values)
    return searched_cv

def train_model(config):
    train_df = pd.read_csv(f'{config.folder}/trainData.csv')
    train_df = train_df.fillna(0)
    test_df = pd.read_csv(f'{config.folder}/testData.csv')
    test_df = test_df.fillna(0)

    train_df = edit_features(train_df)
    test_df = edit_features(test_df)
    all_performances_stats = []

    if config.run_hyperparameter_tuning:
        searched_cv = hpt_job(train_df)
        print('\n All results:')
        print(searched_cv.cv_results_)
        print('\n Best estimator:')
        print(searched_cv.best_estimator_)
        print('\n Best hyperparameters:')
        print(searched_cv.best_params_)
        results = pd.DataFrame(searched_cv.cv_results_)
        results.to_csv('xgb-random-grid-search-results-03q.csv', index=False)
    else:
        for entry in config.optimised_settings.keys():
            print(f'\nFor entry {entry}:\n')

            model_stats = {}

            model = run_training(train_df, entry, config.optimised_settings)
            train_df, train_stats = assess_model(model, train_df, 'train', entry)
            test_df, test_stats = assess_model(model, test_df, 'test', entry)
            model_stats = {**train_stats, **test_stats}

            model_size = save_model(model, config.folder, entry)
            save_importances(model, config.folder, entry)
            model_stats['modelSize'] = model_size
            model_stats['identifier'] = entry

            all_performances_stats.append(model_stats)

        perf_df = pd.DataFrame(all_performances_stats)
        perf_df.to_csv(f'{config.folder}/modelPerformance.csv', index=False)
        train_df.to_csv(f'{config.folder}/trainPreds.csv', index=False)
        test_df.to_csv(f'{config.folder}/testPreds.csv', index=False)
