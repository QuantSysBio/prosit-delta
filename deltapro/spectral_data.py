import json
from math import acos, pi
import multiprocessing
from operator import gt
import os

import numpy as np
import pandas as pd

from deltapro.mgf import process_mgf_file
from deltapro.msp import msp_to_df
# from deltapro.mzml import process_mzml_file
from deltapro.spectral_match import match_prosit_to_observed

def normed_dot_product(true, predicted):
    """ Function to calculate the normalised dot product between the
        true and predicted spectra.
    """
    true_l2_norm = np.linalg.norm(np.array(list(true.values())), ord=2)
    pred_l2_norm = np.linalg.norm(np.array(list(predicted.values())), ord=2)

    if true_l2_norm == 0 and pred_l2_norm == 0:
        return 0.0
    elif true_l2_norm == 0 and pred_l2_norm == 0:
        return 1.0

    product = 0.0
    for ion in true:
        product += (true[ion] * predicted.get(ion, 0.0))

    l2_norm_product = true_l2_norm * pred_l2_norm
    if l2_norm_product > 0:
        product /= l2_norm_product
    product = max(min(product, 1.0), 0.0)
    return product

def calculate_spectral_angle(true, predicted, limit=None):
    """ Function to calculate the spectral angle between the true and predicted
        spectra.
    """
    try:
        if limit is not None:
            new_true = {k:v for k, v in true.items() if k.startswith(limit)}
            new_predicted = {k:v for k, v in predicted.items() if k.startswith(limit)}
            product = normed_dot_product(new_true, new_predicted)
        else:
            product = normed_dot_product(true, predicted)

        spectral_distance = 2*acos(product)/pi

        return 1.0 - spectral_distance
    except:
        return None



def process_spectral_data(config):
    flip_df = pd.read_csv(f'{config.output_folder}/flippedSeqs.csv')
    flip_df['scan'] = flip_df['scan'].apply(lambda x : int(x.split(':')[-1]) if isinstance(x, str) else x)

    all_scans = []
    for scan_file in config.scan_files:
        scans_df = process_mgf_file(scan_file, set(flip_df['scan'].tolist()))
        all_scans.append(scans_df)
    total_scans_df = pd.concat(all_scans)

    flip_df = pd.merge(
        flip_df,
        total_scans_df,
        how='inner',
        on=['source', 'scan']
    )
    list_df = np.array_split(flip_df, 2)

    n_cores = min(config.n_cores, 2)

    results = []
    for chunk_idx, flip_df in enumerate(list_df):
        results.append(process_chunk(flip_df, chunk_idx, config.output_folder, config))

    all_dfs = []
    for entry in results:
        all_dfs.append(pd.read_csv(entry))
    total_flip_df = pd.concat(all_dfs)
    total_flip_df.to_csv(f'{config.output_folder}/spectralData.csv', index=False)
    for entry in results:
        os.remove(entry)

def process_chunk(flip_df, chunk_id, folder, config):
    print(f'Running chunk {chunk_id}, size {flip_df.shape[0]}')
    for idx in range(1, 6):
        msp_df = msp_to_df(f'{folder}/prositPredictions{idx}.msp').rename(
            columns={
                'modified_sequence': f'flip{idx}',
                'Z': 'charge',
                'prositIons': f'flip{idx}PrositIons',
            }
        )
        msp_df[f'flip{idx}'] = msp_df[f'flip{idx}'].apply(lambda x : x.replace('M(ox)', 'm'))

        msp_df = msp_df.drop_duplicates(subset=[f'flip{idx}', 'charge'])
        flip_df = pd.merge(
            flip_df,
            msp_df,
            how='left',
            on=[f'flip{idx}', 'charge']
        )

        flip_df = flip_df.apply(
            lambda x : match_prosit_to_observed(x, f'flip{idx}', f'flip{idx}PrositIons', 0.03, 'Da'),
            axis=1
        )

        flip_df[f'flipSpectralAngle{idx}'] = flip_df.apply(
            lambda x : calculate_spectral_angle(x['prositMatchedIons'], x[f'flip{idx}PrositIons']),
            axis=1
        )

        flip_df = flip_df.drop([f'flip{idx}PrositIons', 'prositMatchedIons'], axis=1)



    msp_df = msp_to_df(f'{folder}/prositPredictions0.msp', with_ce=True).rename(
        columns={
            'modified_sequence': 'peptide',
            'Z': 'charge',
        }
    )
    msp_df = msp_df.drop_duplicates(subset=['peptide', 'charge'])
    msp_df['peptide'] = msp_df['peptide'].apply(lambda x : x.replace('M(ox)', 'm'))

    flip_df = pd.merge(
        flip_df,
        msp_df,
        how='inner',
        on=['peptide', 'charge']
    )

    flip_df = flip_df.apply(
        lambda x : match_prosit_to_observed(x, 'peptide', 'prositIons', 0.03, 'Da'),
        axis=1
    )
    flip_df['spectralAngle'] = flip_df.apply(
        lambda x : calculate_spectral_angle(x['prositMatchedIons'], x['prositIons']),
        axis=1,
    )
    flip_df = flip_df[flip_df['spectralAngle'] > 0.0]

    flip_df['prositIons'] = flip_df['prositIons'].apply(json.dumps)
    flip_df['prositMatchedIons'] = flip_df['prositMatchedIons'].apply(json.dumps)
    flip_df = flip_df[[
        'peptide',
        'charge',
        'spectralAngle',
        'collisionEnergy',
        'matchedCoverage',
        'nMatchedDivFrags',
        'source',
        'scan',
        'flip1',
        'flipInd1',
        'flipYNewIntensity1',
        'flipBNewIntensity1',
        'flipSpectralAngle1',
        'flip2',
        'flipInd2',
        'flipYNewIntensity2',
        'flipBNewIntensity2',
        'flipSpectralAngle2',
        'flip3',
        'flipInd3',
        'flipYNewIntensity3',
        'flipBNewIntensity3',
        'flipSpectralAngle3',
        'flip4',
        'flipInd4',
        'flipYNewIntensity4',
        'flipBNewIntensity4',
        'flipSpectralAngle4',
        'flip5',
        'flipInd5',
        'flipYNewIntensity5',
        'flipBNewIntensity5',
        'flipSpectralAngle5',
        'prositIons',
        'prositMatchedIons',
    ]]
    flip_df.to_csv(f'{folder}/spectralData{chunk_id}.csv', index=False)
    return f'{folder}/spectralData{chunk_id}.csv'
