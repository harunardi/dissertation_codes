import numpy as np

################################################################
case_name = "TASK1_TEST01_1D1G_general"

# Initialization
a = 150
N = 301
dx = (2*a) / (N-1)
x = np.arange(-a, a+dx, dx)
group = 1

# Boundary Conditions
BC = [1, 1] # left, right

################################################################
# Cross Sections
D_0 = 1.341
NUFIS_0 = 2.330E-2
ABS_0 = 2.315E-2

D1 = [1.0*D_0 for _ in range(N)]
ABS1 = [1.0*ABS_0 for _ in range(N)]
NUFIS1 = [1.0*NUFIS_0 for _ in range(N)]
SIGS11 = [0.0 for _ in range(N)]
TOT1 = [x + y for x, y in zip(ABS1, SIGS11)]
D = [D1]
TOT = [TOT1]
NUFIS = [NUFIS1]
SIGS = [SIGS11]
###############################################################
# ILU PRE-CONDITIONER
precond = 0

# Dynamic Parameters
v = 4.13067E5
Beff = 0.00535
l = 0.08510
omega = 2 * np.pi * 1
v = np.ones(N)*v
v = [v]
chi1 = np.zeros(N)+1.0
chi = [chi1]

# Point source
xp = -90

# DYNAMIC CROSS SECTION DEFINITIONS
dSIGS11 = [0.0 for _ in range(N)]
dABS1 = [0.0 for _ in range(N)]
dNUFIS1 = [0.0 for _ in range(N)]
dNUFIS = [dNUFIS1]
dTOT1 = [x + y for x, y in zip(dABS1, dSIGS11)]
dTOT = [dTOT1]
dSIGS = [dSIGS11]

dSOURCE1 = [0.0 for _ in range(N)]
index_S = int((xp + a) / dx)
dSOURCE1[index_S] = 1.0
dSOURCE = dSOURCE1

geom_type = '1D'