from setuptools import setup

setup(
    name='deltapro',
    version=0.2,
    packages=['deltapro'],
    description='Predicting sensitivity of Prosit to permutation.',
    author='John Cormican',
    author_email='john.cormican@mpinat.mpg.de',
    long_description=open('README.md').read(),
    py_modules=[
        'deltapro',
    ],
    entry_points={
        'console_scripts': [
            'deltapro=deltapro.run:main'
        ]
    },
    install_requires=[
        'certifi==2021.10.8',
        'cycler==0.11.0',
        'fonttools==4.37.1',
        'joblib==1.1.0',
        'kiwisolver==1.4.4',
        'logomaker==0.8',
        'matplotlib==3.5.3',
        'numpy==1.23.2',
        'packaging==21.3',
        'pandas==1.4.3',
        'Pillow==9.2.0',
        'plotly==5.10.0',
        'pyparsing==3.0.9',
        'pyteomics==4.5.5',
        'python-dateutil==2.8.2',
        'pytz==2022.2.1',
        'PyYAML==6.0',
        'scikit-learn==1.1.1',
        'scipy==1.9.0',
        'six==1.16.0',
        'tenacity==8.0.1',
        'threadpoolctl==3.1.0',
        'xgboost==1.6.2',
    ],
)