import numpy as np
import matplotlib.pyplot as plt
import json
import time
from scipy.sparse.linalg import spilu, LinearOperator, cg, spsolve, gmres, splu
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#*************************************************************************************
# Load the JSON data from the file
with open('INPUTS/TASK1_TEST10_3DMG_CSTest02_VandV.json', 'r') as f:
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
dy = data['dy']
dz = data['dz']
FLX1_SOL = data['FLX1_SOL']
FLX2_SOL = data['FLX2_SOL']
FLX1_CORESIM = data['FLX1_CORESIM']
FLX2_CORESIM = data['FLX2_CORESIM']

## INITIALIZATION
case_name = "TASK1_TEST10_3DMG_CSTest02_VandV"
I_max = len(D1[0][0]) # N row
J_max = len(D1[0]) # N column
K_max = len(D1)
group = 2

# BC
BC = [3, 3, 3, 3, 3, 3] # N, S, E, W, T, B

# CROSS SECTION DEFINITIONS
TOT1 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
TOT2 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
chi1 = [[[1.0] * I_max for _ in range(J_max)] for _ in range(K_max)]
chi2 = [[[0.0] * I_max for _ in range(J_max)] for _ in range(K_max)]

SIGS12 = REM
SIGS21 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
SIGS11 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
SIGS22 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]

# Perform element-wise addition
for k in range(K_max):
    for j in range(J_max):  # Iterate over columns
        for i in range(I_max):  # Iterate over rows
            TOT1[k][j][i] = ABS1[k][j][i] + SIGS12[k][j][i]
            TOT2[k][j][i] = ABS2[k][j][i] + SIGS21[k][j][i]

TOT = [TOT1, TOT2]
NUFIS = [NUFIS1, NUFIS2]
chi = [chi1, chi2]
D = [D1, D2]

# Reshaping
N = I_max * J_max * K_max
TOT_reshaped = [[None] * N for _ in range(group)]
NUFIS_reshaped = [[None] * N for _ in range(group)]
chi_reshaped = [[None] * N for _ in range(group)]
SIGS12_reshaped = [0.0 for _ in range(N)]
SIGS21_reshaped = [0.0 for _ in range(N)]
SIGS11_reshaped = [0.0 for _ in range(N)]
SIGS22_reshaped = [0.0 for _ in range(N)]
for g in range(group):
    for k in range(K_max):
        for j in range(J_max):
            for i in range(I_max):
                m = k * (I_max * J_max) + j * I_max + i
                chi_reshaped[g][m] = chi[g][k][j][i]
                TOT_reshaped[g][m] = TOT[g][k][j][i]
                NUFIS_reshaped[g][m] = NUFIS[g][k][j][i]
                SIGS11_reshaped[m] = SIGS11[k][j][i]
                SIGS12_reshaped[m] = SIGS12[k][j][i]
                SIGS21_reshaped[m] = SIGS21[k][j][i]
                SIGS22_reshaped[m] = SIGS22[k][j][i]

SIGS_reshaped = [[SIGS11_reshaped, SIGS21_reshaped], [SIGS12_reshaped, SIGS22_reshaped]]

TOT = TOT_reshaped
chi = chi_reshaped
NUFIS = NUFIS_reshaped
S_ADJ = [[0.0] * N for _ in range(group)]
