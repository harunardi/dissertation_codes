# Dissertation Codes Harun Ardiansyah
This repository is the code to perform all analysis done in Harun Ardiansyah's Dissertation and related papers

## How to run the code:
- Go to SRC folder
- For standard forward, adjoint, or noise simulation, use MAIN.py
- To run transfer functions simulation, use TRANSFER_FUNCTION.py
- To run unfolding method, use UNFOLDING_{geom_type}.py
  - geom_type is geometry option: 1D, 2D rectangular grid, 2D hexagonal grid, 3D rectangular grid (coming soon), 3D hexagonal grid (coming soon)

## LIST OF TEST

### 1D Rectangular
- TEST01: 1D 1-group homogeneous test, Forward
- TEST02: CORE SIM+ Test 03, 1D 2-group, Forward, Noise
- TEST03: CORE SIM+ Test 05, 1D 2-group, Forward, Noise
- TEST04: CORE SIM+ Test 07, 1D 2-group, Adjoint

### 2D Rectangular
- TEST05: Serpent 17x17 nodes, 2D 2-group, Forward
- TEST06: CORE SIM+ Test 10, 2D 2-group, Forward, Noise
- TEST07: CORE SIM+ C3 Benchmark, 2D 2-group, Forward, Adjoint, Noise
- TEST08: BIBLIS Benchmark, 2D 2-group, Forward, Adjoint
- TEST09: PWR/MOX Benchmark, 2D 2-group, Forward, Adjoint, Noise

### 3D Rectangular
- TEST10: CORE SIM+ Test 02, 3D 2-group, Forward, Adjoint
- TEST11: CORE SIM+ Test 08, 3D 2-group, Forward
- TEST12: CORE SIM+ Test 09, 3D 2-group, Forward
- TEST13: PWR/MOX Benchmark, 3D 2-group, Forward, Adjoint, Noise
- TEST14: CORE SIM+ Test 14, 3D 2-group, Forward, Adjoint

### 2D Hexagonal
- TEST15: Serpent 3 rings, Forward
- TEST16: Homogeneous Hexagonal Reactor, 2D 2-group, Forward, Adjoint
- TEST17: VVER-440, 2D 2-group, Forward, Adjoint
- TEST18: HTTR, 2D 2-group, Forward, Noise, Adjoint
- TEST19: HTTR, 2D 4-group, Forward
- TEST20: HTTR, 2D 7-group, Forward
- TEST21: HTTR, 2D 14-group, Forward

### 3D Hexagonal
- TEST22: Serpent 3 rings, Forward
- TEST23: VVER-440, 3D 2-group, Forward, Adjoint
- TEST24: HTTR, 3D 2-group, Forward, Noise, Adjoint

### 2D Transfer Function (for space dependences of neutron noise)
- TEST01: CORE SIM+ C3 Benchmark, 2D 2-group, Forward, Noise, Adjoint, Power Perturbations
- TEST02: PWR/MOX Benchmark, 2D 2-group, Forward, Noise, Adjoint, Power Perturbations
- TEST03: HTTR, 2D 2-group, Forward, Noise, Adjoint, Power Perturbations

### Rectangular Noise Models
- TEST04: CORE SIM+ C3 Benchmark AVS, 3D 2-group, Noise Center/Non-center, Absorber of Variable Strength (AVS)
- TEST05: BIBLIS Benchmark AVS, 2D 2-group, Noise Non-center, Absorber of Variable Strength (AVS)
- TEST06: BIBLIS Benchmark FAV, 2D 2-group, Noise Non-center, Fuel Assembly Vibration (FAV)
- TEST07: PWR/MOX Benchmark AVS, 2D 2-group, Noise Non-center, Absorber of Variable Strength (AVS)
- TEST08: PWR/MOX Benchmark AVS/TV, 3D 2-group, Noise Non-center, Absorber of Variable Strength (AVS) and Travelling Vibration
- TEST09: PWR/MOX Benchmark FAV, 2D 2-group, Noise Non-center, Fuel Assembly Vibration (FAV)

### Hexagonal Noise Models
- TEST10: HTTR AVS, 2D 2-group, Noise Center/Non-center, Absorber of Variable Strength (AVS)
- TEST11: HTTR AVS/TV, 3D 2-group, Noise Center/Non-center, Absorber of Variable Strength (AVS) and Travelling Vibration
- TEST12: HTTR FAV, 2D 2-group, Noise Center/Non-center, Fuel Assembly Vibration (FAV)

## NOTE: Required libraries to run the codes
    conda create --name noise numpy scipy matplotlib petsc4py -c conda-forge
    conda install h5py
    conda install -c conda-forge 'petsc=*=complex*' petsc4py

## End Goals
- End goal for 2D is to simulate vibration case
- End goal from 3D is to simulate traveling vibration case