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

#from SRC_ALL.UTILS_GENERAL_3D_RECT import *

# Load the JSON data from the file
with open('INPUTS/TASK1_TEST14_3DMG_CSnoise.json', 'r') as f:
    data = json.load(f)

# Access each matrix using its key
ABS1 = data['ABS1']
ABS2 = data['ABS2']
dABS2 = data['dABS2']
D1 = data['D1']
D2 = data['D2']
NUFIS1 = data['NUFIS1']
NUFIS2 = data['NUFIS2']
REM = data['REM']
dx = data['dx']
dy = data['dy']
dz = data['dz']
FLX1_CORESIM = data['FLX1_CORESIM']
FLX2_CORESIM = data['FLX2_CORESIM']
Beff = data['Beff']
f = data['f']
l = data['l']
v1 = data['v1']
v2 = data['v2']
keff = 1.010459482734583

dFLX1_real = np.array(data['dFLX1_CORESIM_real'])
dFLX1_imag = np.array(data['dFLX1_CORESIM_imag'])
dFLX2_real = np.array(data['dFLX2_CORESIM_real'])
dFLX2_imag = np.array(data['dFLX2_CORESIM_imag'])
dFLX1 = dFLX1_real + 1j * dFLX1_imag
dFLX2 = dFLX2_real + 1j * dFLX2_imag

## INITIALIZATION
case_name = "TASK1_TEST14_3DMG_CSnoise"
I_max = len(D1[0][0]) # N row
J_max = len(D1[0]) # N column
K_max = len(D1)
group = 2
x = np.arange(0, I_max*dx, dx)
y = np.arange(0, J_max*dy, dy)
z = np.arange(0, K_max*dz, dz)

# BC
BC = [3, 3, 3, 3, 3, 3] # N, S, E, W, T, B

# CROSS SECTION DEFINITIONS
TOT1 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
TOT2 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dTOT1 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dTOT2 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dNUFIS1 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dNUFIS2 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
chi1 = [[[1.0] * I_max for _ in range(J_max)] for _ in range(K_max)]
chi2 = [[[0.0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dABS1 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]

SIGS12 = REM
SIGS21 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
SIGS11 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
SIGS22 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]

dSIGS12 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dSIGS21 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dSIGS11 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dSIGS22 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]

dSOURCE1 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dSOURCE2 = [[[0] * I_max for _ in range(J_max)] for _ in range(K_max)]
dSOURCE = [dSOURCE1, dSOURCE2]

# Perform element-wise addition
for k in range(K_max):
    for j in range(J_max):  # Iterate over columns
        for i in range(I_max):  # Iterate over rows
            TOT1[k][j][i] = ABS1[k][j][i] + SIGS12[k][j][i]
            TOT2[k][j][i] = ABS2[k][j][i] + SIGS21[k][j][i]
            dTOT1[k][j][i] = dABS1[k][j][i] + dSIGS12[k][j][i]
            dTOT2[k][j][i] = dABS2[k][j][i] + dSIGS21[k][j][i]

FLX_CORESIM = [FLX1_CORESIM, FLX2_CORESIM]
TOT = [TOT1, TOT2]
dTOT = [dTOT1, dTOT2]
NUFIS = [NUFIS1, NUFIS2]
dNUFIS = [dNUFIS1, dNUFIS2]
chi = [chi1, chi2]
D = [D1, D2]
v1 = [[[v1] * I_max for _ in range(J_max)] for _ in range(K_max)]
v2 = [[[v2] * I_max for _ in range(J_max)] for _ in range(K_max)]
v = [v1, v2]
omega = 2 * np.pi * f

# Reshaping
N = I_max * J_max * K_max
FLX_CORESIM_reshaped = [[None] * N for _ in range(group)]
TOT_reshaped = [[None] * N for _ in range(group)]
dTOT_reshaped = [[None] * N for _ in range(group)]
NUFIS_reshaped = [[None] * N for _ in range(group)]
dNUFIS_reshaped = [[None] * N for _ in range(group)]
dSOURCE_reshaped = [[None] * N for _ in range(group)]
chi_reshaped = [[None] * N for _ in range(group)]
v_reshaped = [[None] * N for _ in range(group)]
SIGS12_reshaped = [0.0 for _ in range(N)]
SIGS21_reshaped = [0.0 for _ in range(N)]
SIGS11_reshaped = [0.0 for _ in range(N)]
SIGS22_reshaped = [0.0 for _ in range(N)]
dSIGS12_reshaped = [0.0 for _ in range(N)]
dSIGS21_reshaped = [0.0 for _ in range(N)]
dSIGS11_reshaped = [0.0 for _ in range(N)]
dSIGS22_reshaped = [0.0 for _ in range(N)]
for g in range(group):
    for k in range(K_max):
        for j in range(J_max):  
            for i in range(I_max):
                m = k * (I_max * J_max) + j * I_max + i
                FLX_CORESIM_reshaped[g][m] = FLX_CORESIM[g][k][j][i]
                TOT_reshaped[g][m] = TOT[g][k][j][i]
                dTOT_reshaped[g][m] = dTOT[g][k][j][i]
                NUFIS_reshaped[g][m] = NUFIS[g][k][j][i]
                dNUFIS_reshaped[g][m] = dNUFIS[g][k][j][i]
                dSOURCE_reshaped[g][m] = dSOURCE[g][k][j][i]
                chi_reshaped[g][m] = chi[g][k][j][i]
                v_reshaped[g][m] = v[g][k][j][i]
                SIGS11_reshaped[m] = SIGS11[k][j][i]
                SIGS12_reshaped[m] = SIGS12[k][j][i]
                SIGS21_reshaped[m] = SIGS21[k][j][i]
                SIGS22_reshaped[m] = SIGS22[k][j][i]
                dSIGS11_reshaped[m] = dSIGS11[k][j][i]
                dSIGS12_reshaped[m] = dSIGS12[k][j][i]
                dSIGS21_reshaped[m] = dSIGS21[k][j][i]
                dSIGS22_reshaped[m] = dSIGS22[k][j][i]

SIGS_reshaped = [[SIGS11_reshaped, SIGS21_reshaped], [SIGS12_reshaped, SIGS22_reshaped]]
dSIGS_reshaped = [[dSIGS11_reshaped, dSIGS21_reshaped], [dSIGS12_reshaped, dSIGS22_reshaped]]

TOT = TOT_reshaped
chi = chi_reshaped
NUFIS = NUFIS_reshaped
dTOT = dTOT_reshaped
v = v_reshaped
dNUFIS = dNUFIS_reshaped
dSOURCE = dSOURCE_reshaped
FLX_CORESIM = FLX_CORESIM_reshaped

#def PHI_matrix(g, N, conv, PHI):
#    max_conv = max(conv)
#    matrix = lil_matrix((g*max_conv, 1))
#    for group in range(g):
#        for n in range(N):
#            matrix[group*max_conv+(conv[n]-1), 0] += PHI[group][n]
#    print("PHI_mat generated")
#    return matrix.toarray().flatten().tolist()
#
#conv = convert_index(D, I_max, J_max, K_max)
#PHI = PHI_matrix(group, N, conv, FLX_CORESIM)

noise_pos = 0
type_noise = 0