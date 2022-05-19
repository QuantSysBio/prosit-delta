# prosit-delta
Predicting the sensitivity of Prosit to amino acid permutaiton.

## Set Up

1) To start with create a new conda environment with python version 3.8:

```
conda create --name deltapro python=3.8
```

2) Activate the environment:

```
conda activate deltapro
```

2) You will then need to install the spire package:

```
python setup.py install
```

4) To check your installation, run

```
deltapro -h
```

## Toy Example

To run a toy example execute the following commands.

1) To generate prositInput files in the example/output folder:

```
deltapro --config_file example/config.yml --pipeline flipSequences
```

2) To save time, we have provided the needed prosit predictions in the output folder. You can then immediately run the preprocessing pipeline:

```
deltapro --config_file example/config.yml --pipeline preprocess
```

3) Finally to train the model and save it as "reg.pkl" in the output folder.

```
deltapro --config_file example/config.yml --pipeline train
```

Note that this model is trained on less than 100 PSMs and is for illustrative purpouses only.

## Execution

As with inSPIRE (https://github.com/QuantSysBio/inSPIRE), this repo is managed using a config file and with different pipelines specified.

```
deltapro --pipeline <choice-of-pipeline> --config_file <path-to-config>
```

### Pipelines

Three pipelines are available "flipSequences", "preprocess", and "train".

| Config Key | Description |
|-------|---------------|
| flipSequences | Randomly selects psoitions where the position of adjacent amino acids will be swapped and runs |
| preprocess | A dictionary mapping the scan files to the collision energy used in each case. |
| train | The number of permutations to be used per PSM. Default is 5. |
| outputFolder | The folder where all output will be written. |

### Config File

The specifications required for the config file are as follows:

| Config Key | Description |
|-------|---------------|
| searchFiles | A list of the search files to be used as training data. |
| scanFiles | A list of mgf or mzML files containing the experimental scan data. |
| collisionEnergies | A dictionary mapping the scan files to the collision energy used in each case. |
| nFlips | The number of permutations to be used per PSM. Default is 5. |
| outputFolder | The folder where all output will be written. |
