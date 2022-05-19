import random
import re

import pandas as pd

def flip_n(df_row, n_flips):
    """ Helper function to flip adjacent amino acids at n randomly chosen positions.

    Parameters
    ----------
    df_row : pd.Series
        A row of a DataFrame containing peptide sequence to be modified.
    n_flips : int
        The number of residue positions to flip.

    Returns
    -------
    df_row : pd.Series
        The row of a DataFrame containing flipped sequences.
    """
    to_be_flipped = [i for i in range(1, len(df_row['peptide']))]

    random.shuffle(to_be_flipped)

    for idx, flipper in enumerate(to_be_flipped):
        if (
            df_row['peptide'][flipper-1] != df_row['peptide'][flipper] and
            idx < n_flips
        ):
            df_row[f'flip{idx+1}'] = (
                df_row['peptide'][:flipper-1] +
                df_row['peptide'][flipper] +
                df_row['peptide'][flipper-1] +
                df_row['peptide'][flipper+1:]
            )
            df_row[f'flipInd{idx+1}'] = flipper

    return df_row

def generate_flipped_data(search_files, n_flips, output_folder, collision_energies):
    """ Function to generate training data with flipped amino acid positions.

    Parameters
    ----------
    search_files : list of str
        A list of PEAKS search result files.
    n_flips : int
        The number of amino acid position flips to make per sequence.
        Recommended value is 5.
    output_folder : str
        The folder where output will be written.
    collision_energies : dict
        A dictionary mapping source files to the collision energy setting used in the MS.
    """
    search_df = pd.concat([pd.read_csv(in_file) for in_file in search_files])

    search_df = search_df[['Source File', 'Scan', 'Peptide', 'Z']].rename(
        columns={
            'Source File': 'source',
            'Scan': 'scan',
            'Peptide': 'peptide',
            'Z': 'charge',
        }
    )
    search_df['source'] = search_df['source'].apply(
        lambda x : x.replace('.mzML', '').replace('.mgf', '').replace('.raw', '')
    )

    search_df['peptide'] = search_df['peptide'].apply(
        lambda x : x.replace('M(+15.99)', 'm').replace('C(+57.02)', 'c')
    )
    search_df = search_df[search_df['peptide'].apply(lambda x : 'C' not in x)]
    search_df['peptide'] = search_df['peptide'].apply(
        lambda x : x.replace('c', 'C')
    )
    search_df = search_df[search_df['peptide'].apply(lambda x : isinstance(x, str))]
    search_df = search_df.apply(lambda x : flip_n(x, n_flips), axis=1)

    flip_cols = []
    for i in range(1, n_flips+1):
        flip_cols.extend([f'flip{i}', f'flipInd{i}'])

    search_df = search_df[[
        'peptide',
        'charge',
        'source',
        'scan',
    ] + flip_cols]
    search_df['collision_energy'] = search_df['source'].apply(lambda x : collision_energies[x])

    search_df.to_csv(f'{output_folder}/flippedSeqs.csv', index=False)

    write_prosit_input(search_df, 'peptide', 'charge', f'{output_folder}/prositInput0.csv')
    for idx in range(1, n_flips+1):
        write_prosit_input(search_df, f'flip{idx}', 'charge', f'{output_folder}/prositInput{idx}.csv', )

def write_prosit_input(gt_df, pep_key, charge_key, out_key):
    prosit_df = gt_df[[pep_key, charge_key, 'collision_energy']].rename(
        columns={
            pep_key: 'modified_sequence',
            charge_key: 'precursor_charge',
        }
    )
    prosit_df = prosit_df.dropna()
    lengths = prosit_df['modified_sequence'].apply(len)
    prosit_df = prosit_df[(lengths >6) & (lengths < 31)]
    prosit_df['modified_sequence'] = prosit_df['modified_sequence'].apply(
        lambda x : x.replace('m', 'M(ox)')
    )
    prosit_df.to_csv(out_key, index=False)
