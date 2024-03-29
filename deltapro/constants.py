
BLOSUM6_1_VALUES = {
    'A': 0.19,
    'C': -1.05,
    'D': 0.01,
    'E': -0.08,
    'F': 0.29,
    'G': 1.19,
    'H': -0.79,
    'I': 0.28,
    'K': 0.1,
    'L': 0.34,
    'M': 0.37,
    'N': 0.83,
    'P': -2.02,
    'Q': -0.08,
    'R': 0.2,
    'S': 0.54,
    'T': 0.38,
    'V': 0.16,
    'W': 0.24,
    'Y': -0.48
}

AA_ENCODING = {
    'A': [0.19, 0, 0, 0],
    'C': [-1.05, 0, 1, 0],
    'D': [0.01, -1, 0, 0],
    'E': [-0.08, -1, 0, 0],
    'F': [0.29, 0, 0, 0],
    'G': [1.19, 0, 1, 0],
    'H': [-0.79, 0.5, 0, 0],
    'I': [0.28, 0, 0, 0],
    'K': [0.1, 1, 0, 0],
    'L': [0.34, 0, 0, 0],
    'M': [0.37, 0, 0, 0],
    'N': [0.83, 0, 0, 1],
    'P': [-2.02, 0, 1, 0],
    'Q': [-0.08, 0, 0, 1],
    'R': [0.2, 1, 0, 0],
    'S': [0.54, 0, 0, 1],
    'T': [0.38, 0, 0, 1],
    'V': [0.16, 0, 0, 0],
    'W': [0.24, 0, 0, 0],
    'Y': [-0.48, 0, 0, 0],
}

OXIDATION_WEIGHT = 15.994915

RESIDUE_WEIGHTS = {
    'A': 71.037114,
    'R': 156.101111, 
    'N': 114.042927, 
    'D': 115.026943, 
    'C': 103.009185 + 57.021464, 
    'E': 129.042593, 
    'Q': 128.058578, 
    'G': 57.021464, 
    'H': 137.058912, 
    'I': 113.084064, 
    'L': 113.084064, 
    'K': 128.094963, 
    'M': 131.040485, 
    'm': 131.040485 + OXIDATION_WEIGHT,
    'F': 147.068414, 
    'P': 97.052764, 
    'S': 87.032028, 
    'T': 101.047679, 
    'W': 186.079313, 
    'Y': 163.06332, 
    'V': 99.068414, 
}

RESIDUE_PROPERTIES = {
    'A': {
        'polarity': 0,
        'hydrophobicity': 1.8,
        'pka': 2.35,
    },
    'C': {
        'polarity': 0,
        'hydrophobicity': 2.5,
        'pka': 1.92,
    },
    'D': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 1.99,
    },
    'E': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 2.10,
    },
    'F': {
        'polarity': 0.0,
        'hydrophobicity': -2.8,
        'pka': 2.20,
    },
    'G': {
        'polarity': 0.0,
        'hydrophobicity': -0.4,
        'pka': 2.35,
    },
    'H': {
        'polarity': 1.0,
        'hydrophobicity': -3.2,
        'pka': 1.80,
    },
    'I': {
        'polarity': 0.0,
        'hydrophobicity': 4.5,
        'pka': 2.32,
    },
    'K': {
        'polarity': 1.0,
        'hydrophobicity': -3.9,
        'pka': 2.16,
    },
    'L': {
        'polarity': 0.0,
        'hydrophobicity': 3.8,
        'pka': 2.33,
    },
    'M': {
        'polarity': 0.0,
        'hydrophobicity': 1.9,
        'pka': 2.13,
    },
    'N': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 2.14,
    },
    'P': {
        'polarity': 0.0,
        'hydrophobicity': -1.6,
        'pka': 1.95,
    },
    'Q': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 2.17,
    },
    'R': {
        'polarity': 1.0,
        'hydrophobicity': -4.5,
        'pka': 1.82,
    },
    'S': {
        'polarity': 1.0,
        'hydrophobicity': -0.8,
        'pka': 2.19,
    },
    'T': {
        'polarity': 1.0,
        'hydrophobicity': -0.7,
        'pka': 2.09,
    },
    'V': {
        'polarity': 0.0,
        'hydrophobicity': 4.2,
        'pka': 2.29,
    },
    'W': {
        'polarity': 0.0,
        'hydrophobicity': -0.9,
        'pka': 2.46,
    },
    'Y': {
        'polarity': 1.0,
        'hydrophobicity': -1.3,
        'pka': 2.20,
    },
}
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


ION_OFFSET = {
    "b": N_TERMINUS - H,
    "y": C_TERMINUS + H,
}
