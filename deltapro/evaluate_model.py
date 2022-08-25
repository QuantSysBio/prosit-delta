from math import comb
import sys
import joblib
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

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
from deltapro.constants import BLOSUM6_1_VALUES

def load_data(folder, title, mod_name):
    combined_df = pd.concat([pd.read_csv(f'{folder}/{tt}Data.csv') for tt in (
        'train', 'test'
    )])
    combined_df.to_csv(f'{folder}/evaluation.csv')


    combined_df['pepLen'] = combined_df['peptide'].apply(len)
    combined_df['nTermDist'] = combined_df['flipInd'].apply(int)
    combined_df['cTermDist'] = combined_df['pepLen'] - combined_df['nTermDist']

    combined_df['cNeighbourBlosum'] = combined_df['cNeighbour'].apply(lambda x : BLOSUM6_1_VALUES.get(x,-5.0))
    combined_df['nNeighbourBlosum'] = combined_df['nNeighbour'].apply(lambda x : BLOSUM6_1_VALUES.get(x, -5.0))
    # mod_name = 'reg15'
    model = joblib.load(f'outputBig/model/{mod_name}.pkl')
    print(mod_name)
    if 'ip' in folder:
        mini_df = combined_df[combined_df['peptide'] == 'RPLVGQDEF']
        mini_df[FEATURE_SET].to_csv('temp.csv')
    combined_df['predictedDiff'] = model.predict(combined_df[FEATURE_SET].values)
    combined_df.to_csv(f'{folder}/predictions.csv', index=False)
    model_stats = {
        f'{title}mae': mean_absolute_error(combined_df[TARGET_VARIABLE], combined_df[f'predictedDiff']),
        f'{title}r2': r2_score(combined_df[TARGET_VARIABLE], combined_df[f'predictedDiff']),
        f'{title}pearson': pearsonr(combined_df[TARGET_VARIABLE], combined_df[f'predictedDiff'])[0],
        f'{title}spearmanr': spearmanr(combined_df[TARGET_VARIABLE], combined_df[f'predictedDiff'])[0],
    }

    return model_stats

if __name__ == '__main__':
    mod_name = sys.argv[1]
    evaluation_folder = sys.argv[2]
    name = sys.argv[3]
    eval_perf = load_data(evaluation_folder, name, mod_name)
    print(eval_perf)

