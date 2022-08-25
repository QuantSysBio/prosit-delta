""" Functions for loading experimental spectra from mzml files.
"""
import numpy as np
import pandas as pd
# from pyopenms import MSExperiment, MzMLFile # pylint: disable-msg=E0611

# from spire.constants import INTENSITIES_KEY, MZS_KEY, SCAN_KEY, SOURCE_KEY


def process_mzml_file(mzml_filename, scan_ids):
    """ Function to process an MzML file to find matches with scan IDs.

    Parameters
    ----------
    mzml_filename : str
        The mzml file from which we are reading.
    scan_ids : list of int
        A list of the scan IDs we require.

    Returns
    -------
    scans_df : pd.DataFrame
        A DataFrame of scan results.
    """
    ion_list = []
    intensities_list = []
    scan_id_list = []
    mzml_filenames = []
    filename = mzml_filename.split('/')[-1].strip('.mzML')

    exp = MSExperiment()
    MzMLFile().load(mzml_filename, exp)
    for spectrum in exp:
        scan_id = int(spectrum.getNativeID().split('scan=')[1])
        if scan_id in scan_ids:
            ion_list.append(np.array([peak.getMZ() for peak in spectrum]))
            intensities_list.append(np.array([peak.getIntensity() for peak in spectrum]))
            scan_id_list.append(scan_id)
            mzml_filenames.append(filename)

    scans_df =  pd.DataFrame(
        {
            'source': pd.Series(mzml_filenames),
            'scan': pd.Series(scan_id_list),
            'Intensities': pd.Series(intensities_list),
            'MZs': pd.Series(ion_list)
        }
    )

    scans_df = scans_df.drop_duplicates(subset=['source', 'scan'])

    return scans_df
