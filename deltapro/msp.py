""" Functions for reading in Prosit predicted spectra in msp format.
"""
import re

import numpy as np
import pandas as pd

CHARGE_KEY = 'charge'
OXIDATION_PREFIX = 'Oxidation@M'
OXIDATION_PREFIX_LEN = len(OXIDATION_PREFIX)
PROSIT_IONS_KEY = 'prositIons'
PROSIT_SEQ_KEY = 'modified_sequence'

def msp_process_sequence_and_charge(line):
    """ Function to extract the name and charge of a sample from
        the relevant line of an msp file.

    Parameters
    ----------
    line : str
        The latest line read from the msp file.

    Returns
    -------
    sequence : str
        The peptide sequence.
    charge : int
        The charge of the sequence.
    """
    regex_match = re.match(r'Name: (.*?)/(\d+)\n', line)
    sequence = regex_match.group(1)
    charge = int(regex_match.group(2))
    return sequence, charge

def process_intensities(ions, intensities):
    """ Function to to combine ion identities and intensities into a single dict.

    Parameters
    ----------
    ions : list of float
        A list of the ion mzs found in the spectrum.
    ions : list of str
        A list of the ion names found in the spectrum.

    Returns
    -------
    matched_ion_intensities : dict
        A dictionary of ion names mapped to their normed intensity.
    """
    l2_norm = np.linalg.norm(np.array(intensities), ord=2)
    normed_intensities = [z/l2_norm for z in intensities]
    matched_ion_intensities = {}
    for ion_info, normed_intensity, in zip (ions, normed_intensities):
        matched_ion_intensities[ion_info] = normed_intensity
    return matched_ion_intensities

def msp_process_peaks(line, msp_file):
    """ Function to extract the ms2 spectrum of a sample from
        the relevant lines of an msp file.

    Parameters
    ----------
    line : str
        The latest line read from the msp file.
    msp_file : file
        The full msp file.

    Returns
    -------
    normed_intensities : dict
        A dictionary of ion names mapped to their normed intensity.
    """
    regex_match = re.match(r'Num peaks: (\d+)\n', line)
    n_peaks = int(regex_match.group(1))

    ions = []
    intensities = []
    for _ in range(n_peaks):
        line = msp_file.readline()
        peak_data = line.strip().split('\t')
        intensity = float(peak_data[1])
        regex_match = re.match(r'(.*?)/(.*?)ppm', peak_data[2].strip('"'))
        ion = regex_match.group(1).strip(')').strip('(')
        if intensity > 0:
            ions.append(ion)
            intensities.append(intensity)

    normed_intensities = process_intensities(ions, intensities)
    return normed_intensities

def get_mods_from_msp_comment(line, sequence):
    """ Function to extract the comments on a sample from
        the relevant line of an msp file.

    Parameters
    ----------
    line : str
        The latest comment line read from the msp file.
    sequence : str
        The latest peptide sequence read from the msp file.

    Returns
    -------
    sequence : str
        The sequence with any modifications added.
    """
    regex_match = re.match(
        r'(.*?) ModString=(.*?)//(.*?)/(.*?)',
        line
    )

    mods = regex_match.group(3)

    if mods:
        mods_list = mods.split('; ')
        mod_seq = ''
        previous_ind = 0
        for mod in mods_list:
            if mod.startswith(OXIDATION_PREFIX):
                pos = int(mod[OXIDATION_PREFIX_LEN:])
                mod_seq += sequence[previous_ind:pos]
                mod_seq += "(ox)"
                previous_ind = pos
        mod_seq += sequence[previous_ind:]
        return mod_seq

    return sequence

def msp_to_df(msp_filename):
    """ Function to process an msp file and extract relevant information
        for training into csv format (tab separated).

    Parameters
    ----------
    msp_filename : str
        The location where the msp file is written.

    Returns
    -------
    ion_df : pd.DataFrame
        The DataFrame with the spectra found in the msp file.
    """
    with open(msp_filename, 'r', encoding='UTF-8') as msp_file:
        line = msp_file.readline()

        peptides = []
        charges = []
        ion_intensities = []
        modified_sequences = []

        while line:
            if line.startswith('Name: '):
                sequence, charge = msp_process_sequence_and_charge(line)

                line = msp_file.readline()
                assert line.startswith('MW: ')

                line = msp_file.readline()
                assert line.startswith('Comment: ')
                modified_sequence = get_mods_from_msp_comment(line, sequence)

                line = msp_file.readline()
                assert line.startswith('Num peaks: ')

                normed_intensities = msp_process_peaks(line, msp_file)

                ion_intensities.append(normed_intensities)
                peptides.append(sequence)
                charges.append(charge)
                modified_sequences.append(modified_sequence)

            line = msp_file.readline()

        ion_df = pd.DataFrame(
            {
                PROSIT_SEQ_KEY: modified_sequences,
                CHARGE_KEY: charges,
                PROSIT_IONS_KEY: ion_intensities,
            }
        )

    return ion_df
