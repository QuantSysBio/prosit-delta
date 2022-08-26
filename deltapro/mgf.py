""" Functions for reading in scans results in mgf format.
"""
import re

import numpy as np
import pandas as pd
from pyteomics import mgf


def process_mgf_file(mgf_filename, scan_ids, scan_file_format=None, source_list=None):
    """ Function to process an mgf file to find matches with scan IDs.

    Parameters
    ----------
    mgf_filename : str
        The mgf file from which we are reading.
    scan_ids : set of int
        A list of the scan IDs we require.
    scan_file_format : str
        The format of the file used.
    source_list : list of str
        A list of source names.

    Returns
    -------
    scans_df : pd.DataFrame
        A DataFrame of scan results.
    """
    matched_scan_ids = []
    matched_intensities = []
    matched_mzs = []
    sources = []
    filename = mgf_filename.split('/')[-1]

    with mgf.read(mgf_filename) as reader:
        for spectrum in reader:
            if scan_file_format is None:
                if 'scans' in spectrum['params']:
                    scan_id = int(spectrum['params']['scans'])
                    source = filename[:-4]
                else:
                    regex_match = re.match(
                        r'(\d+)(.*?)',
                        spectrum['params']['title'].split('scan=')[-1]
                    )
                    scan_id = int(regex_match.group(1))
                    source = filename[:-4]
            else:
                scan_id = int(spectrum['params']['title'].split(' Scan ')[-1].split(' (rt')[0])
                source = source_list[
                    int(spectrum['params']['title'].split(' from file [')[-1].strip(']')) - 1
                ]

            if scan_id in scan_ids:
                sources.append(source)
                matched_scan_ids.append(scan_id)
                matched_intensities.append(np.array(list(spectrum['intensity array'])))
                matched_mzs.append(np.array(list(spectrum['m/z array'])))

    mgf_df = pd.DataFrame(
        {
            'source': pd.Series(sources),
            'scan': pd.Series(matched_scan_ids),
            'Intensities': pd.Series(matched_intensities),
            'MZs': pd.Series(matched_mzs),
        }
    )
    mgf_df = mgf_df.drop_duplicates(subset=['source', 'scan'])

    return mgf_df
