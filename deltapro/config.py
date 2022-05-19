""" Definition of Config class.
"""
import yaml

ALL_CONFIG_KEYS = [
    'searchFiles',
    'collisionEnergies',
    'nFlips',
    'outputFolder',
    'scanFiles',
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