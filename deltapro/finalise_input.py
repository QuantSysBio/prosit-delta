
import pandas as pd
from deltapro.constants import BLOSUM6_1_VALUES, RESIDUE_WEIGHTS, RESIDUE_PROPERTIES, OXIDATION_WEIGHT

def calculate_mass_diff(df_row):
    if df_row['cOxidation'] == 1:
        return abs(RESIDUE_WEIGHTS[df_row['cFlip']] + OXIDATION_WEIGHT - RESIDUE_WEIGHTS[df_row['nFlip']])
    elif df_row['nOxidation'] == 1:
        return abs(RESIDUE_WEIGHTS[df_row['cFlip']] - OXIDATION_WEIGHT - RESIDUE_WEIGHTS[df_row['nFlip']])
    return abs(RESIDUE_WEIGHTS[df_row['cFlip']] - RESIDUE_WEIGHTS[df_row['nFlip']])

def finalise_input(folder):
    
    for tt in ('test', 'train'):
        print(f'{tt.title()} Data...')
        all_dfs = []
        for idx in range(1, 6):
            print(f'Set {idx}')
            feated_df = pd.read_csv(f'{folder}/{tt}FeatedData{idx}.csv')
            feated_df = feated_df.rename(columns={f'flipInd{idx}': 'flipInd'})
            feated_df['specAngleDiff'] = feated_df[f'flipSpectralAngle{idx}'] - feated_df['spectralAngle']

            feated_df['blosumDiff'] = feated_df.apply(
                lambda x : abs(BLOSUM6_1_VALUES[x['cFlip']] - BLOSUM6_1_VALUES[x['nFlip']]),
                axis=1
            )
            feated_df['blosumC'] = feated_df.apply(
                lambda x : BLOSUM6_1_VALUES[x['cFlip']],
                axis=1
            )
            feated_df['blosumN'] = feated_df.apply(
                lambda x : BLOSUM6_1_VALUES[x['nFlip']],
                axis=1
            )
            feated_df['massDiff'] = feated_df.apply(
                lambda x : abs(RESIDUE_WEIGHTS[x['cFlip']] - RESIDUE_WEIGHTS[x['nFlip']]),
                axis=1
            )
            feated_df['hydroDiff'] = feated_df.apply(
                lambda x : abs(RESIDUE_PROPERTIES[x['cFlip']]['hydrophobicity'] - RESIDUE_PROPERTIES[x['nFlip']]['hydrophobicity']),
                axis=1
            )
            feated_df['pkaDiff'] = feated_df.apply(
                lambda x : abs(RESIDUE_PROPERTIES[x['cFlip']]['pka'] - RESIDUE_PROPERTIES[x['nFlip']]['pka']),
                axis=1
            )
            feated_df['polaDiff'] = feated_df.apply(
                lambda x : abs(RESIDUE_PROPERTIES[x['cFlip']]['polarity'] - RESIDUE_PROPERTIES[x['nFlip']]['polarity']),
                axis=1
            )

            all_dfs.append(feated_df[[
                'peptide',
                'source',
                'collisionEnergy',
                'spectralAngle',
                'flipInd',
                'charge',
                'nFlip',
                'cFlip',
                'blosumDiff',
                'massDiff',
                'hydroDiff',
                'pkaDiff',
                'polaDiff',
                'cNeighbour',
                'nNeighbour',
                'blosumN',
                'blosumC',
                'relPos',
                'yIntesAtC',
                'bIntesAtC',
                'cNeighbourOxidation',
                'nNeighbourOxidation',
                'cOxidation',
                'nOxidation',
                'yIntesAtN',
                'bIntesAtN',
                'yIntesAtLoc',
                'bIntesAtLoc',
                'yMatchedIntesAtN',
                'bMatchedIntesAtN',
                'yMatchedIntesAtC',
                'bMatchedIntesAtC',
                'yMatchedIntesAtLoc',
                'bMatchedIntesAtLoc',
                'yErrsAtC',
                'bErrsAtC',
                'yErrsAtN',
                'bErrsAtN',
                'yErrsAtLoc',
                'bErrsAtLoc',
                'flipBNewIntensity',
                'flipYNewIntensity',
                'matchedCoverage',
                'nMatchedDivFrags',
                'specAngleDiff',
            ]])
        total_df = pd.concat(all_dfs)
        total_df.to_csv(
            f'{folder}/{tt}Data.csv',
            index=False,
        )
