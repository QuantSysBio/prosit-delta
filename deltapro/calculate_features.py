import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
import multiprocessing
from multiprocessing import Pool
from deltapro.constants import MZ_ACCURACY
from deltapro.mgf import process_mgf_file
from deltapro.spectral_match import get_ion_mzs, get_matches


def get_intes_at_loc(pep_len, true_ions, loc, letter):
    sum_inte = 0
    if letter == 'y':
        loc = pep_len - loc
    for charge in ['', '^2', '^3']:
        code = letter + str(loc) + charge
        sum_inte += true_ions.get(code, 0.0)
    return sum_inte

def get_err_at_loc(pep_len, matched_ions, prosit_ions, loc, letter):
    sum_err = 0
    if letter == 'y':
        loc = pep_len - loc
    for charge in ['', '^2', '^3']:
        code = letter + str(loc) + charge
        sum_err += abs(matched_ions.get(code, 0.0) - prosit_ions.get(code, 0.0))

    return sum_err

def create_features(df_row, data_ind):
    try:
        peptide = df_row['peptide']
        pep_len = len(peptide)
        index = int(df_row[f'flipInd{data_ind}'])

        prosit_ions = df_row['prositIons']
        prosit_matched_ions = df_row['prositMatchedIons']
        df_row['nFlip'] = peptide[index-1]
        df_row['cFlip'] = peptide[index]
        if df_row['nFlip'] == 'm':
            df_row['nOxidation'] = 1
            df_row['nFlip'] = 'M'
        else:
            df_row['nOxidation'] = 0

        if df_row['cFlip'] == 'm':
            df_row['cOxidation'] = 1
            df_row['cFlip'] = 'M'
        else:
            df_row['cOxidation'] = 0

        if index > 1:
            df_row['nNeighbour'] = peptide[index-2]
        else:
            df_row['nNeighbour'] = 0
            df_row['nNeighbourOxidation'] = 0
            if df_row['nNeighbour'] == 'm':
                df_row['nNeighbour'] = 'M'
                df_row['nNeighbourOxidation'] = 1
            else:
                df_row['cNeighbourOxidation'] = 0
        if index < len(peptide)-1:
            df_row['cNeighbour'] = peptide[index+1]
            if df_row['cNeighbour'] == 'm':
                df_row['cNeighbour'] = 'M'
                df_row['cNeighbourOxidation'] = 1
            else:
                df_row['cNeighbourOxidation'] = 0

        else:
            df_row['cNeighbour'] = 0
            df_row['cNeighbourOxidation'] = 0

        df_row['relPos'] = index/len(peptide)

        df_row['yIntesAtN'] = get_intes_at_loc(pep_len, prosit_ions, index-1, 'y')
        df_row['bIntesAtN'] = get_intes_at_loc(pep_len, prosit_ions, index-1, 'b')
        df_row['yIntesAtLoc'] = get_intes_at_loc(pep_len, prosit_ions, index, 'y')
        df_row['bIntesAtLoc'] = get_intes_at_loc(pep_len, prosit_ions, index, 'b')
        df_row['yIntesAtC'] = get_intes_at_loc(pep_len, prosit_ions, index+1, 'y')
        df_row['bIntesAtC'] = get_intes_at_loc(pep_len, prosit_ions, index+1, 'b')

        df_row['yMatchedIntesAtN'] = get_intes_at_loc(pep_len, prosit_matched_ions, index-1, 'y')
        df_row['bMatchedIntesAtN'] = get_intes_at_loc(pep_len, prosit_matched_ions, index-1, 'b')
        df_row['yMatchedIntesAtLoc'] = get_intes_at_loc(pep_len, prosit_matched_ions, index, 'y')
        df_row['bMatchedIntesAtLoc'] = get_intes_at_loc(pep_len, prosit_matched_ions, index, 'b')
        df_row['yMatchedIntesAtC'] = get_intes_at_loc(pep_len, prosit_matched_ions, index+1, 'y')
        df_row['bMatchedIntesAtC'] = get_intes_at_loc(pep_len, prosit_matched_ions, index+1, 'b')

        df_row['yErrsAtN'] = get_err_at_loc(pep_len, prosit_matched_ions, prosit_ions, index-1, 'y')
        df_row['bErrsAtN'] = get_err_at_loc(pep_len, prosit_matched_ions, prosit_ions, index-1, 'b')
        df_row['yErrsAtLoc'] = get_err_at_loc(pep_len, prosit_matched_ions, prosit_ions, index, 'y')
        df_row['bErrsAtLoc'] = get_err_at_loc(pep_len, prosit_matched_ions, prosit_ions, index, 'b')
        df_row['yErrsAtC'] = get_err_at_loc(pep_len, prosit_matched_ions, prosit_ions, index+1, 'y')
        df_row['bErrsAtC'] = get_err_at_loc(pep_len, prosit_matched_ions, prosit_ions, index+1, 'b')

        df_row['flipYNewIntensity'] = df_row[f'flipYNewIntensity{data_ind}']
        df_row['flipBNewIntensity'] = df_row[f'flipBNewIntensity{data_ind}']

    except Exception as e:
        # print(e)
        df_row['nFlip'] = 'x'
    return df_row

def stratify(x):
    if x < 0.2:
        return 0
    if x < 0.4:
        return 1
    if x < 0.6:
        return 2
    if x < 0.8:
        return 3
    return 4

def calculate_features(folder, config):
    """ Function to compute all of the input feature for the deltapro predictor.
    """
    spec_df = pd.read_csv(f'{folder}/spectralData.csv')
    spec_df['prositIons'] = spec_df['prositIons'].apply(json.loads)
    spec_df['prositMatchedIons'] = spec_df['prositMatchedIons'].apply(json.loads)
    spec_df['saStrata'] = spec_df['spectralAngle'].apply(stratify)
    # train, test = train_test_split(spec_df, test_size=0.2, stratify=spec_df['saStrata'])
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=42)
    split = splitter.split(spec_df, groups=spec_df['peptide'])
    train_inds, test_inds = next(split)
    train = spec_df.iloc[train_inds]
    test = spec_df.iloc[test_inds]
    print('split data')
    print(train.shape, test.shape)

    # func_args = [(train, test, idx, folder, config.scan_files) for idx in range(1, 6)]
    # pool = Pool(processes=1)
    # results = pool.starmap(process_chunk, func_args)
    for idx in range(1, 6):
        process_chunk(train, test, idx, folder, config.scan_files)

def process_chunk(train, test, idx, folder, scan_files):
    print(idx)

    print(train.shape)

    train = train.apply(
        lambda x : create_features(x, idx),
        axis=1,
    )
    train = train[train['nFlip'] != 'x']
    train.drop(['prositIons', 'prositMatchedIons'], axis=1).to_csv(f'{folder}/trainFeatedData{idx}.csv', index=False)

    test = test.apply(
        lambda x : create_features(x, idx),
        axis=1,
    )
    test = test[test['nFlip'] != 'x']
    test.drop(['prositIons', 'prositMatchedIons'], axis=1).to_csv(f'{folder}/testFeatedData{idx}.csv', index=False)
