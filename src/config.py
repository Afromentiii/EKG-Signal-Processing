# Klasy binarne pod funkcje straty squared_hinge (SVM): -1 (norma), +1 (arytmia)
BINARY_LABEL_MAPPING = {
    'N': -1, 'L': -1, 'R': -1, 'e': -1, 'j': -1,
    'V': 1, 'E': 1, 'A': 1, 'a': 1, 'J': 1,
    'S': 1, 'F': 1, 'Q': 1, '/': 1
}

# Standard AAMI EC57 - podzial na 5 superklas
AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']

AAMI_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

AAMI_CLASS_NAMES = {
    'N': 'N (Norma / Bloki odnog)',
    'S': 'S (Nadkomorowe ektopowe)',
    'V': 'V (Komorowe ektopowe)',
    'F': 'F (Pobudzenia zlane)',
    'Q': 'Q (Stymulator / Nieznane)',
}

# Standardowe parametry sygnalu MIT-BIH (fs=360 Hz) oraz pasmo filtracji EKG (0.5 - 40 Hz)
DEFAULT_WINDOW_SIZE = 300
DEFAULT_FS = 360
DEFAULT_LOWCUT = 0.5
DEFAULT_HIGHCUT = 40.0
DEFAULT_ORDER = 4
DEFAULT_L2_REG = 0.01

# Parametry asymetrycznego okna QRS (-50 ms do +100 ms)
DEFAULT_ASYM_PRE_MS = 50
DEFAULT_ASYM_POST_MS = 100
DEFAULT_ASYM_WINDOW_SIZE = 54  # 18 probek pre-R + 36 probek post-R przy 360 Hz

SAMPLE_RECORDS = ['100', '101', '102', '104']

SAMPLE_PATIENT_GROUPS = {
    'Standard (bez stymulatora)': ['100', '101'],
    'Pacemaker (ze stymulatorem)': ['102', '104'],
}

ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

PACEMAKER_RECORDS = ['102', '104', '107', '217']
STANDARD_RECORDS = [r for r in ALL_RECORDS if r not in PACEMAKER_RECORDS]

FULL_PATIENT_GROUPS = {
    'Standard (bez stymulatora)': STANDARD_RECORDS,
    'Pacemaker (ze stymulatorem)': PACEMAKER_RECORDS,
}

# Podzial miedzyosobniczy de Chazala (2004) - zbior DS1 (trening) i DS2 (test)
DE_CHAZAL_DS1 = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]
DE_CHAZAL_DS2 = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]
