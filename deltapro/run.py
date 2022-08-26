from argparse import ArgumentParser
import random

import numpy as np
from deltapro.analyse import analyse

from deltapro.config import Config
from deltapro.calculate_features import calculate_features
from deltapro.finalise_input import finalise_input
from deltapro.flip_residues import generate_flipped_data
from deltapro.spectral_data import process_spectral_data
from deltapro.train_model import train_model


random.seed(42)
np.random.seed(42)

PIPELINE_OPTIONS = [
    'flipSequences',
    'preprocess',
    'train',
    'analyse',
]

def get_arguments():
    """ Function to collect command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = ArgumentParser(description='Deltapro predictor of Prosit Spectral Angle Deltas.')

    parser.add_argument(
        '--config_file',
        help='Config file to be read from.',
        type=str
    )
    parser.add_argument(
        '--pipeline',
        choices=PIPELINE_OPTIONS,
        help='What pipeline do you want to run?',
    )

    return parser.parse_args()

def main():
    """ Function to orchestrate running of each spi-screen pipeline.
    """
    args = get_arguments()
    config = Config(args.config_file)
    config.validate(args.pipeline)

    if args.pipeline == 'flipSequences':
        generate_flipped_data(
            config.search_files,
            config.n_flips,
            config.output_folder,
            config.collision_energies,
        )

    if args.pipeline == 'preprocess':
        process_spectral_data(
            config,
        )
        calculate_features(
            config.output_folder, config
        )
        finalise_input(
            config.output_folder,
        )

    if args.pipeline == 'train':
        train_model(
            config,
        )

    if args.pipeline == 'analyse':
        analyse(
            config,
        )

if __name__ == '__main__':
    main()
