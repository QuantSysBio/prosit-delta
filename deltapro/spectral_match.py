import numpy as np

from deltapro.constants import OXIDATION_WEIGHT


KNOWN_PTM_WEIGHTS = {
    'Deamidated (NQ)': 0.984016,
    'Oxidation (M)': 15.994915,
    'Acetyl (N-term)': 42.010565,
    'Phospho (Y)': 79.966331,
    'Phospho (ST)': 79.966331,
    'Carbamidomethyl (C)': 57.021464,
}

PROTON = 1.007276466622
ELECTRON = 0.00054858
H = 1.007825035
C = 12.0
O = 15.99491463
N = 14.003074

N_TERMINUS = H
C_TERMINUS = O + H
CO = C + O
CHO = C + H + O
NH2 = N + H * 2
H2O = H * 2 + O
NH3 = N + H * 3
LOSS_WEIGHTS = {"":0, "NH3": NH3, "H2O": H2O}
MZ_ACCURACY = 0.03

RESIDUE_WEIGHTS = {
    "A": 71.037114,
    "R": 156.101111, 
    "N": 114.042927, 
    "D": 115.026943, 
    "C": 103.009185 + 57.021464, 
    "E": 129.042593, 
    "Q": 128.058578, 
    "G": 57.021464, 
    "H": 137.058912, 
    "I": 113.084064, 
    "L": 113.084064, 
    "K": 128.094963, 
    "M": 131.040485, 
    "F": 147.068414, 
    "P": 97.052764, 
    "S": 87.032028, 
    "T": 101.047679, 
    "W": 186.079313, 
    "Y": 163.06332, 
    "V": 99.068414, 
}

ION_OFFSET = {
    "b": N_TERMINUS - H,
    "y": C_TERMINUS + H,
}

PEAKS_FEATURES = [
    # 'peaks_score',
    'mass_diff',
    'abs_mass_diff',
    'charges',
    'pep_len',
    'var_mods',
    'prosit_unknown_mods',
]

def get_matches(
        all_prosit_ions,
        prosit_intensities,
        observed_mzs,
        observed_intensities,
        mz_accuracy
    ):
    """ Function to find all the matched mz measuremnets between the
        predicted MS2 spectrum of a peptide and an observed MS2 spectra.

    Parameters
    ----------
    expected_mzs : np.array
        An array of the mz of expected fragments that could be generated
        for the peptide.
    prosit_intensities : np.array
        The predicted intensities from prosit. These are used to 
    observed_mzs : np.array
        An array of the mz of the observed fragments from the MS spectrum.
    observed_intensities : np.array
        An array of the intensities observed for the corresponding observed
        mzs.
    mz_accuracy : float
        The accuracy of the m/z measurement for the observations.

    Returns
    -------
    matches : np.array
        A 2d array of shape (len(observed_mzs), n ion types considered).
        If there is no match between the observed mz and a given ion type
        the value of the corresponding array entry is 0. Otherwise it will
        be the fragment index.
    """    
    # Loop over observed fragments matching them to all possible prosit ions.
    final_intensities = {}

    for ion_type_loss in all_prosit_ions:
        for fragment_idx in range(len(all_prosit_ions[ion_type_loss])):
            fragment = all_prosit_ions[ion_type_loss][fragment_idx]
            matched_mz_ind = np.argmin(
                np.abs(observed_mzs - fragment)
            )
            if abs(observed_mzs[matched_mz_ind] - fragment) < mz_accuracy:
                ion_code = ion_type_loss[0] + str(fragment_idx+1) + ion_type_loss[1:]
                if ion_code in prosit_intensities:
                    final_intensities[ion_code] = observed_intensities[matched_mz_ind]

    l2_norm_ion_final = np.linalg.norm(np.array(list(final_intensities.values())), ord=2)
    for ion_code in final_intensities:
        final_intensities[ion_code] = final_intensities[ion_code]/l2_norm_ion_final
    return final_intensities


def get_ion_mzs(sequence, ptm_id_weights, modifications=None):
    """ Function to the get mz's for all the ions predicted by prosit.

    Parameters
    ----------
    sequence : str
        The peptide sequence for which we require molecular weights.
    modifications : str
        A string of the ptms for the sequence which will alter
        the potential mzs.

    Returns
    -------
    all_possible_ions : dict
        A dictionary of all the mzs of all possible b and y ions
        that could be produced.
    """
    sub_seq_mass, total_precusor_weight = compute_potential_mzs(
        sequence=sequence,
        modifications=modifications,
        reverse=False,
        ptm_id_weights=ptm_id_weights
    )

    rev_sub_seq_mass, _ = compute_potential_mzs(
        sequence=sequence,
        modifications=modifications,
        reverse=True,
        ptm_id_weights=ptm_id_weights
    )

    all_possible_ions = {}
    all_possible_ions['b'] = ION_OFFSET['b'] + sub_seq_mass + (1*PROTON)
    all_possible_ions['b^2'] = (ION_OFFSET['b'] + sub_seq_mass + (2*PROTON))/2
    all_possible_ions['b^3'] = (ION_OFFSET['b'] + sub_seq_mass + (3*PROTON))/3
    all_possible_ions['y'] = ION_OFFSET['y'] + rev_sub_seq_mass + PROTON
    all_possible_ions['y^2'] = (ION_OFFSET['y'] + rev_sub_seq_mass + (2*PROTON))/2
    all_possible_ions['y^3'] = (ION_OFFSET['y'] + rev_sub_seq_mass + (3*PROTON))/3

    return all_possible_ions, total_precusor_weight

def compute_potential_mzs(sequence, modifications, reverse, ptm_id_weights):
    """ Function to compute the molecular weights of potential fragments
        generated from a peptide (y & b ions, charges 1,2, or 3, and H2O
        or O2 losses).

    Parameters
    ----------
    sequence : str
        The peptide sequence for which we require molecular weights.
    modifications : str
        A string of the ptms for the sequence which will alter
        the potential mzs.
    reverse : bool
        Whether we are getting fragment mzs in the forward direction
        (eg for b ions), or backward direction (eg. for y ions).

    Returns
    -------
    mzs : np.array of floats
        An array of all the possible mzs that coule be observed in
        the MS2 spectrum of a sequence.
    """
    try:
        sequence_length = len(sequence)
    except Exception as e:
        raise ValueError(f'on {sequence}, {modifications}')
    n_fragments = sequence_length - 1
    mzs = np.empty(n_fragments)
    modifications = str(modifications)

    if modifications and modifications != 'nan':
        ptms_list = modifications.split(".")
        if reverse:
            ptm_start = int(ptms_list[2])
            ptm_end = int(ptms_list[0])
        else:
            ptm_start = int(ptms_list[0])
            ptm_end = int(ptms_list[2])

        mods_list = [int(mod) for mod in ptms_list[1]]
    else:
        mods_list = [0] * sequence_length
        ptm_start = 0
        ptm_end = 0

    if reverse:
        sequence = sequence[::-1]
        mods_list = mods_list[::-1]


    tracking_mw = ptm_id_weights[ptm_start]

    for idx in range(n_fragments):
        tracking_mw += RESIDUE_WEIGHTS[sequence[idx]]
        if mods_list[idx]:
            tracking_mw += ptm_id_weights[mods_list[idx]]

        mzs[idx] = tracking_mw

    tracking_mw += RESIDUE_WEIGHTS[sequence[n_fragments]]
    if mods_list[n_fragments]:
        tracking_mw += ptm_id_weights[mods_list[n_fragments]]
    if ptm_end:
        tracking_mw += ptm_id_weights[ptm_end]

    return mzs, tracking_mw


def match_prosit_to_observed(df_row, peptide_key, prosit_key):
    """ Function to extract the ion intensities from the true spectra which match
    """
    try:
        sequence = df_row[peptide_key]
        ion_mzs = df_row['MZs']
        intensities = np.array(df_row['Intensities'])
        prosit_preds = df_row[prosit_key]
        if 'm' in sequence:
            ptm_seq = '0.'
            for char in sequence:
                if char == 'm':
                    ptm_seq += '1'
                else:
                    ptm_seq += '0'

            ptm_seq += '.0'
            un_mod_seq = sequence.replace('m', 'M')


            potential_ion_mzs, _ = get_ion_mzs(
                un_mod_seq,
                {0: 0.0, 1: OXIDATION_WEIGHT},
                ptm_seq
            )
        else:
            potential_ion_mzs, _ = get_ion_mzs(
                sequence,
                {0: 0.0},
                'nan'
            )


        matched_intensities = get_matches(
            potential_ion_mzs,
            prosit_preds,
            ion_mzs,
            intensities,
            MZ_ACCURACY
        )

        df_row['prositMatchedIons'] = matched_intensities
    except Exception as e:
        if isinstance(sequence, str) and 'm' in sequence:
            print(sequence)
            # raise e
        df_row['prositMatchedIons'] = None
    return df_row
