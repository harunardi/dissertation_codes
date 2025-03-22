import numpy as np
import json

# INITIALIZATION
case_name = "TASK1_TEST04_1DMG_CSTest07"
a = 150
N = 301
dx = (2*a) / (N-1)
x = np.arange(-a, a+dx, dx)
group = 2

# Load the JSON data from the file
with open('INPUTS/TASK1_TEST04_1DMG_CSTest07.json', 'r') as f:
    data = json.load(f)

# Access each matrix using its key
ABS1 = data['ABS1']
ABS2 = data['ABS2']
D1 = data['D1']
D2 = data['D2']
NUFIS1 = data['NUFIS1']
NUFIS2 = data['NUFIS2']
REM = data['REM']
dx = data['dx']
FLX1_CORESIM = data['FLX1_CORESIM']
FLX2_CORESIM = data['FLX2_CORESIM']
FLX1_ANA = data['FLX1_ANA']
FLX2_ANA = data['FLX2_ANA']
keff_coresim = data['keff_coresim']

# BC
BC = [3,3]

# CROSS SECTION DEFINITIONS
SIGS12 = REM
SIGS21 = [0.0 for _ in range(N)]
SIGS11 = [0.0 for _ in range(N)]
SIGS22 = [0.0 for _ in range(N)]
SIGS = [[SIGS11, SIGS21], [SIGS12, SIGS22]]
TOT1 = [x + y for x, y in zip(ABS1, SIGS12)]
TOT2 = [x + y for x, y in zip(ABS2, SIGS21)]
D = [D1, D2]
TOT = [TOT1, TOT2]
NUFIS = [NUFIS1, NUFIS2]
chi1 = [1.0 for _ in range(N)]
chi2 = [0.0 for _ in range(N)]
chi = [chi1, chi2]
dSOURCE1 = [0.0 for _ in range(N)]
dSOURCE2 = [0.0 for _ in range(N)]
dSOURCE = [dSOURCE1, dSOURCE2]

## ILU PRE-CONDITIONER
precond = 0
geom_type = '1D'
