import numpy as np

from deltapro.constants import RESIDUE_WEIGHTS, ION_OFFSET, PROTON, MZ_ACCURACY






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
    if l2_norm_ion_final:
        for ion_code in final_intensities:
            final_intensities[ion_code] = final_intensities[ion_code]/l2_norm_ion_final
    return final_intensities, l2_norm_ion_final


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
        reverse=False,
    )

    rev_sub_seq_mass, _ = compute_potential_mzs(
        sequence=sequence,
        reverse=True,
    )

    all_possible_ions = {}
    all_possible_ions['b'] = ION_OFFSET['b'] + sub_seq_mass + (1*PROTON)
    all_possible_ions['b^2'] = (ION_OFFSET['b'] + sub_seq_mass + (2*PROTON))/2
    all_possible_ions['b^3'] = (ION_OFFSET['b'] + sub_seq_mass + (3*PROTON))/3
    all_possible_ions['y'] = ION_OFFSET['y'] + rev_sub_seq_mass + PROTON
    all_possible_ions['y^2'] = (ION_OFFSET['y'] + rev_sub_seq_mass + (2*PROTON))/2
    all_possible_ions['y^3'] = (ION_OFFSET['y'] + rev_sub_seq_mass + (3*PROTON))/3

    return all_possible_ions, total_precusor_weight

def compute_potential_mzs(sequence, reverse):
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
    n_fragments = len(sequence) - 1
    mzs = np.empty(n_fragments)


    if reverse:
        sequence = sequence[::-1]

    tracking_mw = 0.0

    for idx in range(n_fragments):
        tracking_mw += RESIDUE_WEIGHTS[sequence[idx]]

        mzs[idx] = tracking_mw

    tracking_mw += RESIDUE_WEIGHTS[sequence[n_fragments]]

    return mzs, tracking_mw


def match_prosit_to_observed(df_row, peptide_key, prosit_key, mz_accuracy, mz_units):
    """ Function to extract the ion intensities from the true spectra which match
    """
    try:
        sequence = df_row[peptide_key]
        pep_len = len(sequence)
        n_frags = pep_len - 1

        ion_mzs = df_row['MZs']
        intensities = np.array(df_row['Intensities'])
        prosit_preds = df_row[prosit_key]

        potential_ion_mzs, _ = get_ion_mzs(
            sequence,
            {0: 0.0},
            'nan'
        )


        matched_intensities, l2_norm = get_matches(
            potential_ion_mzs,
            prosit_preds,
            ion_mzs,
            intensities,
            MZ_ACCURACY
        )
        
        if peptide_key.startswith('flip'):
            flip_no = int(peptide_key[-1])
            
            b_new, y_new = calculate_intes_at_new_loc(df_row, flip_no)
            if b_new > 0:
                df_row[f'flipBNewIntensity{flip_no}'] = b_new/(l2_norm+b_new)
            else:
                df_row[f'flipBNewIntensity{flip_no}'] = 0.0
            if y_new > 0:
                df_row[f'flipYNewIntensity{flip_no}'] = y_new/(l2_norm+y_new)
            else:
                df_row[f'flipYNewIntensity{flip_no}'] = 0.0
        else:
            y_inds = [
                int(ion.split('^')[0][1:]) for ion in matched_intensities if ion.startswith('y')
            ]
            b_inds = [
                int(ion.split('^')[0][1:]) for ion in matched_intensities if ion.startswith('b')
            ]
            y_rev_inds = [pep_len-x for x in y_inds]
            df_row['nMatchedDivFrags'] = len(matched_intensities)/n_frags
            df_row['matchedCoverage'] = len(set(b_inds+y_rev_inds))/n_frags
        df_row['prositMatchedIons'] = matched_intensities
    except Exception as e:
        df_row['prositMatchedIons'] = None
        if peptide_key.startswith('flip'):
            flip_no = int(peptide_key[-1])
            df_row[f'flipBNewIntensity{flip_no}'] = None
            df_row[f'flipYNewIntensity{flip_no}'] = None
    return df_row


def calculate_intes_at_new_loc(df_row, flip_no):
    flip_idx = df_row[f'flipInd{flip_no}']
    try:
        flip_idx = int(flip_idx)
    except:
        return None
    flip_peptide = df_row[f'flip{flip_no}']
    b_frag = flip_peptide[:flip_idx]
    b_frag_mass = sum([RESIDUE_WEIGHTS[res_char] for res_char in b_frag])
    y_frag = flip_peptide[flip_idx:]
    y_frag_mass = sum([RESIDUE_WEIGHTS[res_char] for res_char in y_frag])

    b_new_matched_inte = 0.0
    for charge in range(1, min(4, df_row['charge']+1)):
        ion = (ION_OFFSET['b'] + b_frag_mass + (charge*PROTON))/charge
        matched_mz_ind = np.argmin(
            np.abs(df_row['MZs'] - ion)
        )
        if abs(df_row['MZs'][matched_mz_ind] - ion) < 0.035:
            b_new_matched_inte += df_row['Intensities'][matched_mz_ind]

    y_new_matched_inte = 0.0
    for charge in range(1, min(4, df_row['charge']+1)):
        ion = (ION_OFFSET['y'] + y_frag_mass + (charge*PROTON))/charge
        matched_mz_ind = np.argmin(
            np.abs(df_row['MZs'] - ion)
        )
        if abs(df_row['MZs'][matched_mz_ind] - ion) < 0.035:
            y_new_matched_inte += df_row['Intensities'][matched_mz_ind]

    return b_new_matched_inte, y_new_matched_inte
