""" Definition of Config class.
"""
import os

import yaml

ALL_CONFIG_KEYS = [
    'bestModel',
    'searchFiles',
    'collisionEnergies',
    'nFlips',
    'outputFolder',
    'scanFiles',
    'nCores',
    'tuneHyperparameters'
]

class Config:
    """ Holder for configuration of the spi-screen pipeline.
    """
    def __init__(self, config_file):
        """ Initialise Config object.
        """
        with open(config_file, 'r', encoding='UTF-8') as stream:
            config_dict = yaml.safe_load(stream)

        for config_key in config_dict:
            if config_key not in ALL_CONFIG_KEYS:
                raise ValueError(f'Unrecognised key {config_key} found in config file.')

        self._load_data(config_dict)

    def _load_data(self, config_dict):
        """ Function to load data.
        """
        self.output_folder = config_dict.get('outputFolder')
        self.search_files = config_dict.get('searchFiles')
        self.scan_files = config_dict.get('scanFiles')
        self.n_flips = config_dict.get('nFlips')
        self.collision_energies = config_dict.get('collisionEnergies')
        self.best_model = config_dict.get('bestModel')
        self.n_cores = config_dict.get('nCores', 1)
        self.tune_hyperparameters = config_dict.get('tuneHyperparameters', False)
        self.optimised_settings = config_dict.get(
            'optimisedSettings',
            {
                'default': {
                    'min_child_weight': 2,
                    'max_depth': 16,
                    'learning_rate': 0.15,
                    'gamma': 0.1,
                    'colsample_bytree': 0.9,
                }
            }
        )
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(f'{self.output_folder}/importances'):
            os.makedirs(f'{self.output_folder}/importances')
        if not os.path.exists(f'{self.output_folder}/model'):
            os.makedirs(f'{self.output_folder}/model')

    def validate(self):
        """ Check that the appropriate Config values have been set for the executed pipeline.
        """
        if self.output_folder is None:
            raise ValueError(
                'You must specify outputFolder in the config.'
            )

        if self.output_folder is None:
            raise ValueError(
                'You must specify searchFiles in the config.'
            )

        if self.output_folder is None:
            raise ValueError(
                'You must specify scanFiles in the config.'
            )

        if self.output_folder is None:
            raise ValueError(
                'You must specify nFlips in the config.'
            )

        if self.output_folder is None:
            raise ValueError(
                'You must specify collisionEnergies in the config.'
            )