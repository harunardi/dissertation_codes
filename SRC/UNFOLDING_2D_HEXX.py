import numpy as np
import json
import time
import os
import sys
from scipy.integrate import trapezoid
import scipy.linalg
from itertools import combinations
from petsc4py import PETSc

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

start_time = time.time()

from UTILS import Utils
from MATRIX_BUILDER import *
from METHODS import *
from POSTPROCESS import PostProcessor
from SOLVERFACTORY import SolverFactory

#######################################################################################################
# INPUTS
original_sys_path = sys.path.copy()
sys.path.append('../')

#from INPUTS.TASK3_TEST03a_2DTriMG_HTTR_1SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST03b_2DTriMG_HTTR_2SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST03c_2DTriMG_HTTR_3SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST03d_2DTriMG_HTTR_4SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST03e_2DTriMG_HTTR_5SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST03f_2DTriMG_HTTR_6SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST03g_2DTriMG_HTTR_1SRC_21DET_AVS import *
#from INPUTS.TASK3_TEST03h_2DTriMG_HTTR_1SRC_10DET_AVS import *
#from INPUTS.TASK3_TEST03i_2DTriMG_HTTR_1SRC_5DET_AVS import *
#from INPUTS.TASK3_TEST03j_2DTriMG_HTTR_1SRC_1DET_AVS import *
from INPUTS.TASK3_TEST03k_2DTriMG_HTTR_LVL2_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST03l_2DTriMG_HTTR_LVL2_FAV_NONCENTER import *

# Restore the original sys.path
sys.path = original_sys_path

#######################################################################################################
def main():
    start_time = time.time()

    output_dir = f'../OUTPUTS/{input_name}'
    global level

##### Forward Simulation
    solver_type = 'forward'
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}', exist_ok=True)
    conv_hexx = convert_2D_hexx(I_max, J_max, D)
    conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
    conv_tri_array = np.array(conv_tri)
    max_conv = max(conv_tri)
    conv_neighbor, tri_indices, x, y, all_triangles = calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level)

    matrix_builder = MatrixBuilderForward2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
    M, F_FORWARD = matrix_builder.build_forward_matrices()

    solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F_FORWARD, h, precond, tol=1E-10)
    keff, PHI_temp = solver.solve()
    PHI, PHI_reshaped, PHI_temp_reshaped = PostProcessor.postprocess_power2DHexx(PHI_temp, conv_tri, group, N_hexx)

    output = {"keff": keff.real}
    for g in range(len(PHI_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### Adjoint Simulation
    solver_type = 'adjoint'
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}', exist_ok=True)

    matrix_builder = MatrixBuilderAdjoint2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
    M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

    solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F_ADJOINT, h, precond, tol=1E-10)
    keff, PHI_ADJ_temp = solver.solve()
    PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_temp_reshaped = PostProcessor.postprocess_power2DHexx(PHI_ADJ_temp, conv_tri, group, N_hexx)

    output = {"keff": keff.real}
    for g in range(len(PHI_ADJ_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### Noise Simulation
    solver_type = 'noise'
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}', exist_ok=True)

    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
        forward_output = json.load(json_file)
    keff = forward_output["keff"]
    PHI_all = []
    for i in range(group):
        phi_key = f"PHI{i+1}_FORWARD"
        PHI_all.append(forward_output[phi_key])

    PHI = np.zeros(max(conv_tri) * group)
    for g in range(group):
        PHI_indices = g * max(conv_tri) + (conv_tri_array - 1)
        PHI[PHI_indices] = PHI_all[g]

    # Noise Input Manipulation
    dTOT_hexx = expand_XS_hexx(group, J_max, I_max, dTOT, level)
    dSIGS_hexx = expand_SIGS_hexx(group, J_max, I_max, dSIGS_reshaped, level)
    chi_hexx = expand_XS_hexx(group, J_max, I_max, chi, level)
    dNUFIS_hexx = expand_XS_hexx(group, J_max, I_max, dNUFIS, level)
    if noise_section == 1:
        # Collect all non-zero indices of dTOT_hexx for each group
        for g in range(group):
            for n in range(N_hexx):
                if dTOT_hexx[g][n] != 0:
                    noise_tri_index = n//(6 * (4 ** (level - 1))) * (6 * (4 ** (level - 1))) + 3
                    if n != noise_tri_index:
                        dTOT_hexx[g][n] = 0
    else:
        pass
    if type_noise == 'FVX' or type_noise == 'FAV':
        if level != 4:
            print('Vibrating Assembly type noise only works if level = 4. Changing level to 4')
            level = 4

    hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)
    triangle_neighbors_global = find_triangle_neighbors_2D(all_triangles, precision=6)

    matrix_builder = MatrixBuilderNoise2DHexx(group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise)
    M, dS = matrix_builder.build_noise_matrices()

    solver = SolverFactory.get_solver_fixed2DHexx(solver_type, group, conv_tri, M, dS, PHI, precond, tol=1e-10)

    dPHI_temp = solver.solve()
    dPHI, dPHI_reshaped, dPHI_temp_reshaped = PostProcessor.postprocess_fixed2DHexx(dPHI_temp, conv_tri, group, N_hexx)

    output = {}
    for g in range(len(dPHI_reshaped)):
        dPHI_groupname = f'dPHI{g + 1}'
        dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
        output[dPHI_groupname] = dPHI_list

    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

#### 01. Green's Function Generation
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_01_GENERATE', exist_ok=True)
    matrix_builder = MatrixBuilderNoise2DHexx(group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise)
    M, dS = matrix_builder.build_noise_matrices()

    M_petsc = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data), comm=PETSc.COMM_WORLD)
    M_petsc.assemble()

    # PETSc Solver (KSP) and Preconditioner (PC)
    ksp = PETSc.KSP().create()
    ksp.setOperators(M_petsc)
    ksp.setType(PETSc.KSP.Type.GMRES)

    pc = ksp.getPC()
    if precond == 0:
        print(f'Solving using Sparse Solver')
        pc.setType(PETSc.PC.Type.NONE)
    elif precond == 1:
        print(f'Solving using ILU')
        pc.setType(PETSc.PC.Type.ILU)
        print(f'ILU Preconditioner Done')
    elif precond == 2:
        print('Solving using LU Decomposition')
        pc.setType(PETSc.PC.Type.LU)
        print(f'LU Preconditioner Done')

    # Solver tolerances
    ksp.setTolerances(rtol=1e-10, max_it=5000)

    G_sol_all = np.ones(group*N_hexx, dtype=complex)
    G_sol_temp = np.ones(group*max_conv, dtype=complex)
    G_matrix = np.zeros((group * max_conv, group * max_conv), dtype=complex)
    
    for g in range(group):
        for n in range(N_hexx):
            if conv_tri[n] != 0:
                dS = [0] * (group * max_conv)
                dS[g*max_conv+(conv_tri[n]-1)] = 1  # Set the relevant entry to 1
                dS_petsc = PETSc.Vec().createWithArray(dS)
                dS_petsc.assemble()

                errdPHI = 1
                tol = 1E-10
                iter = 0

                while errdPHI > tol:
                    G_sol_tempold = np.copy(G_sol_temp)
                    G_sol_temp_petsc = PETSc.Vec().createWithArray(G_sol_temp)

                    # Solve the linear system using PETSc KSP
                    ksp.solve(dS_petsc, G_sol_temp_petsc)

                    # Get result back into NumPy array
                    G_sol_temp = G_sol_temp_petsc.getArray()

                    errdPHI = np.max(np.abs(G_sol_temp - G_sol_tempold) / (np.abs(G_sol_temp) + 1E-20))

                for gp in range(group):
                    for m in range(N_hexx):
                        G_sol_all[gp * N_hexx + m] = G_sol_temp[gp * max_conv + (conv_tri[m] - 1)]
                G_sol_reshape = np.reshape(G_sol_all, (group, N_hexx))
                G_matrix[:, g*max_conv+(conv_tri[n]-1)] = G_sol_temp.flatten()  # Assign solution to row
                
                # OUTPUT
                output = {}
                for gp in range(group):
                    G_sol_groupname = f'G{g+1}{gp+1}'
                    G_sol_list = [{"real": x.real, "imaginary": x.imag} for x in G_sol_reshape[gp]]
                    output[G_sol_groupname] = G_sol_list

               # Save output to HDF5 file
                hdf5_filename = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_01_GENERATE/Green_g{g+1}_n{n+1}.h5'
                save_output_hdf5(hdf5_filename, output)
                print(f'Generated Green Function for group = {g + 1}, N = {n+1}')

##### 02. Solve
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_02_SOLVE', exist_ok=True)
    output_SOLVE = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_02_SOLVE/{case_name}'

    with h5py.File(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_02_SOLVE/{case_name}_G_matrix_full.h5', 'w') as hf:
        hf.create_dataset('G_matrix', data=G_matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_full')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_full')
    plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_02_SOLVE/{case_name}_G_matrix_full.png')

    matrix_builder = MatrixBuilderNoise2DHexx(group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise)
    M, dS = matrix_builder.build_noise_matrices()
    S = dS.dot(PHI)

    # Iterate over group pairs to compute dPHI
    dPHI_temp_SOLVE = np.zeros((group * max_conv), dtype=complex)
    for i in range(group):
        for j in range(group):
            # Extract the relevant blocks from G_matrix and S
            G_block = G_matrix[i*max_conv:(i+1)*max_conv, j*max_conv:(j+1)*max_conv]
            S_block = S[j*max_conv:(j+1)*max_conv]
        
            # Perform the matrix-vector multiplication for the Green's function
            dPHI_temp_SOLVE[i*max_conv:(i+1)*max_conv] += np.dot(G_block, S_block)

    non_zero_indices = np.nonzero(conv_tri)[0]
    dPHI_temp_indices = conv_tri_array[non_zero_indices] - 1
    dPHI_SOLVE = np.zeros((group * N_hexx), dtype=complex)
    S_all = np.zeros((group * N_hexx), dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max_conv
        dPHI_SOLVE[g * N_hexx + non_zero_indices] = dPHI_temp_SOLVE[dPHI_temp_start + dPHI_temp_indices]
        S_all[g * N_hexx + non_zero_indices] = S[dPHI_temp_start + dPHI_temp_indices]
        for n in range(N_hexx):
            if conv_tri[n] == 0:
                dPHI_SOLVE[g*N_hexx+n] = np.nan
                S_all[g*N_hexx+n] = np.nan
    dPHI_SOLVE_reshaped = np.reshape(dPHI_SOLVE, (group, N_hexx))
    S_all_reshaped = np.reshape(S_all, (group, N_hexx))

    # OUTPUT
    print(f'Generating JSON output')
    output = {}
    for g in range(group):
        dPHI_SOLVE_groupname = f'dPHI{g+1}'
        dPHI_SOLVE_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_SOLVE_reshaped[g]]
        output[dPHI_SOLVE_groupname] = dPHI_SOLVE_list

        S_groupname = f'S{g+1}'
        S_list = [{"real": x.real, "imaginary": x.imag} for x in S_all_reshaped[g]]
        output[S_groupname] = S_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_02_SOLVE/{case_name}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    dPHI_temp_SOLVE_reshaped = np.reshape(dPHI_temp_SOLVE, (group, max_conv))
    S_reshaped = np.reshape(S, (group, max_conv))
    for g in range(group):
        plot_triangular_general(dPHI_temp_SOLVE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx (Reference) Magnitude', case_name=case_name, output_dir=output_SOLVE, process_data="magnitude")
        plot_triangular_general(dPHI_temp_SOLVE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx (Reference) Phase', case_name=case_name, output_dir=output_SOLVE, process_data="phase")
        plot_triangular_general(S_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='S', title=f'2D Plot of S{g+1} Hexx (Reference) Magnitude', case_name=case_name, output_dir=output_SOLVE, process_data="magnitude")
        plot_triangular_general(S_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='S', title=f'2D Plot of S{g+1} Hexx (Reference) Phase', case_name=case_name, output_dir=output_SOLVE, process_data="phase")

    # UNFOLDING
    G_inverse = scipy.linalg.inv(G_matrix)
    dS_unfold_temp_SOLVE = np.dot(G_inverse, dPHI_temp_SOLVE)
    dS_unfold_SOLVE = np.zeros((group * N_hexx), dtype=complex)  # Assuming N >= max_conv

    # POSTPROCESS
    print(f'Postprocessing to appropriate dPHI')
    non_zero_indices = np.nonzero(conv_tri)[0]
    dS_unfold_temp_indices = conv_tri_array[non_zero_indices] - 1

    for g in range(group):
        dS_unfold_temp_start = g * max_conv
        dS_unfold_SOLVE[g * N_hexx + non_zero_indices] = dS_unfold_temp_SOLVE[dS_unfold_temp_start + dS_unfold_temp_indices]
        for n in range(N):
            if conv_tri[n] == 0:
                dS_unfold_SOLVE[g*N_hexx+n] = np.nan
    dS_unfold_SOLVE_reshaped = np.reshape(dS_unfold_SOLVE,(group,N_hexx))

    # OUTPUT
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_SOLVE_groupname = f'dS_unfold{g+1}'
        dS_unfold_SOLVE_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_SOLVE_reshaped[g]]
        output[dS_unfold_SOLVE_groupname] = dS_unfold_SOLVE_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_02_SOLVE/{case_name}_dS_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    dS_unfold_temp_SOLVE_reshaped = np.reshape(dS_unfold_temp_SOLVE, (group, max_conv))

    for g in range(group):
        plot_triangular_general(dS_unfold_temp_SOLVE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS', title=f'2D Plot of dS{g+1}_SOLVE Hexx Magnitude', case_name=case_name, output_dir=output_SOLVE, process_data="magnitude")
        plot_triangular_general(dS_unfold_temp_SOLVE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS', title=f'2D Plot of dS{g+1}_SOLVE Hexx Phase', case_name=case_name, output_dir=output_SOLVE, process_data="phase")

    # Calculate error and compare
    diff_S1 = np.abs(np.array(dS_unfold_SOLVE_reshaped[0]) - np.array(S_all_reshaped[0]))
    diff_S2 = np.abs(np.array(dS_unfold_SOLVE_reshaped[1]) - np.array(S_all_reshaped[1]))
    diff_S = [[diff_S1], [diff_S2]]
    diff_S_array = np.array(diff_S)
    diff_S_reshaped = diff_S_array.reshape(group, N_hexx)
    diff_S_temp_all = []
    for g in range(group):
        for n in range(N_hexx):
            m = g * N_hexx + n
            if conv_tri[n] != 0:
                diff_S_temp_all.append(diff_S_reshaped[g][n])
    diff_S_temp_array = np.array(diff_S_temp_all)
    diff_S_temp_reshaped = diff_S_temp_array.reshape(group, max(conv_tri))

    for g in range(group):
        plot_triangular_general(diff_S_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S', title=f'2D Plot of diff_S{g+1} Hexx Magnitude', case_name=case_name, output_dir=output_SOLVE, process_data="magnitude")
        plot_triangular_general(diff_S_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S', title=f'2D Plot of diff_S{g+1} Hexx Phase', case_name=case_name, output_dir=output_SOLVE, process_data="phase")

##### 03. INVERT
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT', exist_ok=True)
    output_INVERT = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT/{case_name}'

    # --------------- MAP DETECTOR -------------------
    # Expand the map detector
    p = 6 * (4 ** (level - 1))
    map_detector_hexx = [0] * I_max * J_max * p
    map_detector_temp = np.reshape(map_detector, (J_max, I_max))
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            for k in range(p):
                map_detector_hexx[m * p + k] = map_detector_temp[j][i]

    map_detector_conv = []
    for n in range(N_hexx):
        if conv_tri[n] != 0:
            map_detector_conv.append(map_detector_hexx[n])

    print(f'Number of known position for each group is {map_detector_conv.count(1)} of {len(map_detector_conv)}')

    for m in range(len(map_detector_hexx)):
        if map_detector_hexx[m] == 9:
            map_detector_hexx[m] = np.nan

    map_detector_hexx_plot = np.array(map_detector_hexx)
    map_detector_conv_plot = np.array(map_detector_conv)
    plot_triangular_general(map_detector_conv_plot, x, y, tri_indices, g+1, cmap='viridis', varname='map_detector', title=f'2D Plot of map_detector_hexx', case_name=case_name, output_dir=output_INVERT, process_data="magnitude")

    centroids = []
    for idx, tri in enumerate(all_triangles):
        tri_coords = [v for v in tri]  # Explicitly unpack triangle vertices
        centroids.append([
            sum(v[0] for v in tri_coords) / 3,
            sum(v[1] for v in tri_coords) / 3
        ])

    centroids_round = []
    for n in range(len(all_triangles)):
        tri_coords = [round_vertex(v) for v in all_triangles[n]]
        centroids_round.append([
            sum(v[0] for v in tri_coords) / 3,
            sum(v[1] for v in tri_coords) / 3
        ])

    # --------------- MANIPULATE dPHI -------------------
    # Define zeroed dPHI as dPHI_temp_zero and dPHI_zero
    dPHI_temp_meas = dPHI_temp.copy() # 1D list, size (group * max_conv)
    for g in range(group):
        for n in range(N_hexx):
            if map_detector_hexx[n] == 0:
                idx = g * max_conv + (conv_tri[n]-1)
                dPHI_temp_meas[idx] = 0

    # Plot dPHI_zero_reshaped
    dPHI_meas_reshaped = np.reshape(dPHI_temp_meas, (group, max_conv))

    for g in range(group):
        plot_triangular_general(dPHI_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Magnitude', case_name=case_name, output_dir=output_INVERT, process_data="magnitude")
        plot_triangular_general(dPHI_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Phase', case_name=case_name, output_dir=output_INVERT, process_data="phase")

    # --------------- INTERPOLATE dPHI -------------------
#    dPHI_interp = interpolate_2D_hexx_rbf(dPHI_meas_reshaped, group, J_max, I_max, level, conv_tri,map_detector_conv, all_triangles, centroids, centroids_round)
    for g in range(group):
        known_coords = []
        known_values = []
        zero_coords = []

        for n in range(len(all_triangles)):
            tri_coords = [round_vertex(v) for v in all_triangles[n]]
            centroid = (
                sum(v[0] for v in tri_coords) / 3,
                sum(v[1] for v in tri_coords) / 3
            )

            if map_detector_conv[n] == 1:
                known_coords.append(centroid)
                known_values.append(dPHI_meas_reshaped[g][n])
            elif map_detector_conv[n] == 0:
                zero_coords.append(centroid)

    zero_coord_to_index = {}
    for idx, tri in enumerate(all_triangles):
        tri_coords = [v for v in tri]
        centroid = (
            sum(v[0] for v in tri_coords) / 3,
            sum(v[1] for v in tri_coords) / 3
        )
    zero_coord_to_index[centroid] = idx  # Map centroid to index

    dPHI_interp = interpolate_2D_hexx_rbf(dPHI_meas_reshaped, group, conv_tri, known_coords, known_values, zero_coords, all_triangles, zero_coord_to_index)
    dPHI_temp_interp = np.zeros((group * max_conv), dtype=complex)
    for g in range(group):
        for n in range(max_conv):
            if map_detector_conv[n] == 1:
                dPHI_interp[g][n] = dPHI_meas_reshaped[g][n]
            dPHI_temp_interp[g * max_conv + n] = dPHI_interp[g][n]

    for g in range(group):
        plot_triangular_general(dPHI_interp[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_interp', title=f'2D Plot of dPHI{g+1}_interp Hexx Magnitude', case_name=case_name, output_dir=output_INVERT, process_data="magnitude")
        plot_triangular_general(dPHI_interp[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_interp', title=f'2D Plot of dPHI{g+1}_interp Hexx Phase', case_name=case_name, output_dir=output_INVERT, process_data="phase")

    # Initialize dPHI_all with zeros (or appropriate default value)
    dPHI_interp_all = np.zeros((group, N_hexx), dtype=complex)

    # Map values from dPHI_temp back to dPHI_all
    for g in range(group):
        for n in range(N_hexx):
            if conv_tri[n] != 0:  # Only map non-zero elements
                dPHI_interp_all[g][n] = dPHI_interp[g][conv_tri[n] - 1]
            else:
                dPHI_interp_all[g][n] = np.nan

    # OUTPUT
    print(f'Generating JSON output')
    output = {}
    for g in range(group):
        dPHI_groupname = f'dPHI{g+1}'
        dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_interp_all[g]]
        output[dPHI_groupname] = dPHI_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT/{case_name}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    diff_flx1 = np.abs((np.array(dPHI_interp[0]) - np.array(dPHI_temp_reshaped[0]))/(np.array(dPHI_temp_reshaped[0]) + 1E-10)) * 100
    diff_flx2 = np.abs((np.array(dPHI_interp[1]) - np.array(dPHI_temp_reshaped[1]))/(np.array(dPHI_temp_reshaped[1]) + 1E-10)) * 100
    diff_flx = [[diff_flx1], [diff_flx2]]
    diff_dPHI_interp_array = np.array(diff_flx)
    diff_dPHI_interp_reshaped = diff_dPHI_interp_array.reshape(group,max_conv)

    for g in range(group):
        plot_triangular_general(diff_dPHI_interp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_dPHI_interp', title=f'2D Plot of diff_dPHI{g+1}_interp Hexx Magnitude', case_name=case_name, output_dir=output_INVERT, process_data="magnitude")
        plot_triangular_general(diff_dPHI_interp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_dPHI_interp', title=f'2D Plot of diff_dPHI{g+1}_interp Hexx Phase', case_name=case_name, output_dir=output_INVERT, process_data="phase")

    # --------------- LOAD GREEN'S FUNCTION -------------------
    # Plot G_matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_full')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_full')
    plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT/{case_name}_G_matrix_full.png')

    # --------------- INTERPOLATE GREEN'S FUNCTION -------------------
    # Delete G_matrix_full at unknown position (column-wise at specific row)
    G_matrix_meas = G_matrix.copy()
    for g in range(group):
        for n in range(max_conv):
            if map_detector_conv[n] == 0:
                G_matrix_meas[g * max_conv + n, :] = 0 # Zeroing a column instead of a row

    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix_meas.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_meas')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_meas')
    plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT/{case_name}_G_matrix_meas.png')

    # Interpolate rows of the Green's function
    G_matrix_interp = G_matrix_meas
    G_matrix_interp_cols = np.zeros((group * max_conv, group * max_conv), dtype=complex) #np.full((group * N_hexx, group * N_hexx), np.nan, dtype=complex)
    for g in range(group):
        for n in range(max_conv):
            G_mat_interp_temp = G_matrix_interp[:, g * max_conv + n]  # Extract a row
            print(f'Interpolating G_mat_interp_temp group {g+1}, position {n+1}')
            G_mat_interp_cols = interpolate_2D_hexx_rbf(G_mat_interp_temp, group, conv_tri, known_coords, known_values, zero_coords, all_triangles, zero_coord_to_index) # Perform interpolation on the column
            G_matrix_interp_cols[:, g * max_conv + n] = G_mat_interp_cols  # Assign back to the row

    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix_interp_cols.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_interp_cols')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_interp_cols')
    plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT/{case_name}_G_matrix_interp_cols.png')

    # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
    print(f'Solve for dS using Direct Method')
    G_mat_interp_inverse = scipy.linalg.pinv(G_matrix_interp_cols)

    # Plot G_matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(G_mat_interp_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_mat_interp_inverse')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_mat_interp_inverse')
    plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT/{case_name}_G_mat_interp_inverse.png')

    # UNFOLD ALL INTERPOLATED
    dS_unfold_INVERT_temp = np.dot(G_mat_interp_inverse, dPHI_temp_interp)
    dS_unfold_INVERT = np.zeros((group* N_hexx), dtype=complex)

    # POSTPROCESS
    print(f'Postprocessing to appropriate dPHI')
    non_zero_conv = np.nonzero(conv_tri)[0]
    dS_unfold_temp_indices = conv_tri_array[non_zero_conv] - 1

    for g in range(group):
        dS_unfold_temp_start = g * max(conv_tri)
        dS_unfold_INVERT[g * N_hexx + non_zero_conv] = dS_unfold_INVERT_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
        for n in range(N_hexx):
            if conv_tri[n] == 0:
                dS_unfold_INVERT[g*N_hexx+n] = np.nan

    dS_unfold_INVERT_reshaped = np.reshape(dS_unfold_INVERT,(group,N_hexx))
    dS_unfold_INVERT_temp_reshaped = np.reshape(dS_unfold_INVERT_temp,(group,max_conv))

    for g in range(group):
        plot_triangular_general(dS_unfold_INVERT_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_unfold_INVERT', title=f'2D Plot of dS{g+1}_unfold_INVERT Hexx Magnitude', case_name=case_name, output_dir=output_INVERT, process_data="magnitude")
        plot_triangular_general(dS_unfold_INVERT_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_unfold_INVERT', title=f'2D Plot of dS{g+1}_unfold_INVERT Hexx Phase', case_name=case_name, output_dir=output_INVERT, process_data="phase")

    # OUTPUT
    print(f'Generating JSON output for dS')
    output_direct1 = {}
    for g in range(group):
        dS_unfold_direct_groupname = f'dS_unfold{g+1}'
        dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_INVERT_reshaped[g]]
        output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_03_INVERT/{case_name}_dS_unfold_INVERT_output.json', 'w') as json_file:
        json.dump(output_direct1, json_file, indent=4)

    # Calculate error and compare
    diff_S1_INVERT = (np.abs(np.array(dS_unfold_INVERT_temp_reshaped[0]) - np.array(S_reshaped[0])) / (np.abs(np.array(S_reshaped[0])) + 1E-20)) * 100
    diff_S2_INVERT = (np.abs(np.array(dS_unfold_INVERT_temp_reshaped[1]) - np.array(S_reshaped[1])) / (np.abs(np.array(S_reshaped[0])) + 1E-20)) * 100
    diff_S_INVERT = [[diff_S1_INVERT], [diff_S2_INVERT]]
    diff_S_INVERT_array = np.array(diff_S_INVERT)
    diff_S_INVERT_reshaped = diff_S_INVERT_array.reshape(group, max_conv)

    for g in range(group):
        plot_triangular_general(diff_S_INVERT_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_dS_INVERT', title=f'2D Plot of diff_dS{g+1}_INVERT Hexx Magnitude', case_name=case_name, output_dir=output_INVERT, process_data="magnitude")
        plot_triangular_general(diff_S_INVERT_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_dS_INVERT', title=f'2D Plot of diff_dS{g+1}_INVERT Hexx Phase', case_name=case_name, output_dir=output_INVERT, process_data="phase")

##### 04. ZONE
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE', exist_ok=True)
    output_ZONE = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}'

    # --------------- MAP DETECTOR -------------------
    # Expand the map detector
    p = 6 * (4 ** (level - 1))
    map_detector_hexx = [0] * I_max * J_max * p
    map_detector_temp = np.reshape(map_detector, (J_max, I_max))
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            for k in range(p):
                map_detector_hexx[m * p + k] = map_detector_temp[j][i]

    map_detector_conv = []
    for n in range(N_hexx):
        if conv_tri[n] != 0:
            map_detector_conv.append(map_detector_hexx[n])

    print(f'Number of known position for each group is {map_detector_conv.count(1)} of {len(map_detector_conv)}')

    for m in range(len(map_detector_hexx)):
        if map_detector_hexx[m] == 9:
            map_detector_hexx[m] = np.nan

    map_detector_hexx_plot = np.array(map_detector_hexx)
    map_detector_conv_plot = np.array(map_detector_conv)
    plot_triangular_general(map_detector_conv_plot, x, y, tri_indices, g+1, cmap='viridis', varname='map_detector', title=f'2D Plot of map_detector_hexx', case_name=case_name, output_dir=output_ZONE, process_data="magnitude")

    map_zone_hexx = [0] * I_max * J_max * p
    map_zone_temp = np.reshape(map_zone, (J_max, I_max))
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            for k in range(p):
                map_zone_hexx[m * p + k] = map_zone_temp[j][i]

    for m in range(len(map_zone_hexx)):
        if map_zone_hexx[m] == 9:
            map_zone_hexx[m] = np.nan

    map_zone_conv = np.zeros((max_conv))
    for n in range(N_hexx):
        if conv_tri[n] != 0:
            map_zone_conv[conv_tri[n] - 1] =  map_zone_hexx[n]

    map_zone_hexx_plot = np.array(map_zone_hexx)
    map_zone_conv_plot = np.array(map_zone_conv)
    plot_triangular_general(map_zone_conv_plot, x, y, tri_indices, g+1, cmap='viridis', varname='map_zone', title=f'2D Plot of map_zone', case_name=case_name, output_dir=output_ZONE, process_data="magnitude")

    zone_length = np.zeros(int(max(map_zone_conv)))
    for z in range(int(max(map_zone_conv))):
        for n in range(max_conv):
            if map_zone_conv[n] == z + 1:
                zone_length[z] += 1

    # --------------- MANIPULATE dPHI -------------------
    # Define zeroed dPHI as dPHI_temp_zero and dPHI_zero
    dPHI_temp_meas = dPHI_temp.copy() # 1D list, size (group * max_conv)
    for g in range(group):
        for n in range(N_hexx):
            if map_detector_hexx[n] == 0:
                idx = g * max_conv + (conv_tri[n]-1)
                dPHI_temp_meas[idx] = 0

    # Plot dPHI_zero_reshaped
    dPHI_meas_reshaped = np.reshape(dPHI_temp_meas, (group, max_conv))

    for g in range(group):
        plot_triangular_general(dPHI_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Magnitude', case_name=case_name, output_dir=output_ZONE, process_data="magnitude")
        plot_triangular_general(dPHI_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Phase', case_name=case_name, output_dir=output_ZONE, process_data="phase")

    # --------------- DIVIDE dPHI TO ZONES -------------------
    dPHI_temp_meas_zone = np.zeros((int(max(map_zone_conv)), group * max_conv), dtype=complex)
    for g in range(group):
        for n in range(max_conv):
            zone_index = int(map_zone_conv[n] - 1)
            dPHI_temp_meas_zone[zone_index][g * max_conv + n] = dPHI_temp_meas[g * max_conv + n]

    filename = f"{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_dPHI_temp_meas_zone.txt"
    with open(filename, "w") as f:
        for zone_index, zone_data in enumerate(dPHI_temp_meas_zone):
            f.write(f"Zone {zone_index + 1}:\n")
            for value in zone_data:
                f.write(f"{value.real:.6e}+{value.imag:.6e}j \n")
            f.write("\n\n")  # Add a blank line between zones

    # --------------- MANIPULATE GREEN'S FUNCTION -------------------
    # Plot G_matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_full')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_full')
    plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_G_matrix_full.png')

    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix_meas.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_meas')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_meas')
    plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_G_matrix_meas.png')

    ###################################################################################################
    # --------------- DIVIDE GREEN'S FUNCTION TO ZONES -------------------
    G_matrix_rows_zone = np.zeros((int(max(map_zone_conv)), group * max_conv, group * max_conv), dtype=complex)
    for g1 in range(group):
        for n1 in range(max_conv):
            for g2 in range(group):
                for n2 in range(max_conv):
                    zone_index = int(map_zone_conv[n2] - 1)
                    G_matrix_rows_zone[zone_index][g1 * max_conv + n1][g2 * max_conv + n2] = G_matrix_meas[g1 * max_conv + n1, g2 * max_conv + n2]

    for z in range(int(max(map_zone_conv))):
        plt.figure(figsize=(8, 6))
        plt.imshow(G_matrix_rows_zone[z].real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_matrix_rows_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_matrix_rows_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_G_matrix_rows_zone{z}.png')

    ###################################################################################################
    # --------------- SOLVE FOR EACH ZONE -------------------
    dS_unfold_ZONE_temp = dPHI_temp_meas.copy()
    for g in range(group):
        for n in range(max_conv):
            dS_unfold_ZONE_temp[g * max_conv + n] = map_zone_conv[n]

    for z in range(int(max(map_zone_conv))):
        G_zone_matrix = G_matrix_rows_zone[z]
        non_zero_cols = ~np.all(G_zone_matrix == 0, axis=0)
        G_zone_mat = G_zone_matrix[:, non_zero_cols]

        plt.figure(figsize=(8, 6))
        plt.imshow(G_zone_mat.real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_matrix_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_matrix_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_G_matrix_zone{z}.png')

        G_zone_square = []
        for g in range(group):
            for n in range(len(map_zone_conv)):
                if map_zone_conv[n] == z + 1:
                    G_zone_square.append(G_zone_mat[g * max_conv + n, :])

        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(G_zone_square), cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_zone_square_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_zone_square_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_G_zone_square_zone{z}.png')

        # Extract the current zone noise flux
        zone_vector = dPHI_temp_meas_zone[z]
        zone_vector_new = []
        for g in range(group):
            for n in range(max_conv):
                if map_zone_conv[n] == z + 1:
                    zone_vector_new.append(zone_vector[n])

        zone_vector_new = np.array(zone_vector_new)

        # Inverse the G_matrix_rows_zone
        zone_matrix_inverse = scipy.linalg.pinv(G_zone_square)

        plt.figure(figsize=(8, 6))
        plt.imshow(zone_matrix_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_inverse_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_inverse_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_G_inverse_zone{z}.png')

        # Direct Solve
        dS_ZONE = []
        dS_ZONE = np.dot(zone_matrix_inverse, zone_vector_new)
        np.savetxt(f"{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_dS_zone{z+1}.txt", dS_ZONE)
        print(f'Zone {z+1}: dS_zone length = {len(dS_ZONE)}')

        dS_ZONE_index = 0
        for i in range(len(dS_unfold_ZONE_temp)):
            if dS_unfold_ZONE_temp[i] == z+1:
                dS_unfold_ZONE_temp[i] = dS_ZONE[dS_ZONE_index]
                dS_ZONE_index += 1

        filename = f"{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_dS_temp_meas_zone{z+1}.txt"
        with open(filename, "w") as f:
            for zone_index, zone_data in enumerate(dS_unfold_ZONE_temp):
                f.write(f"{zone_data.real:.6e}+{zone_data.imag:.6e}j \n")
            f.write("\n\n")  # Add a blank line between zones

    np.savetxt(f"{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_dS_unfold_zone_temp.txt", dS_unfold_ZONE_temp)

    # POSTPROCESS
    dS_unfold_ZONE = np.zeros((group* N_hexx), dtype=complex)
    print(f'Postprocessing to appropriate dPHI')
    non_zero_conv = np.nonzero(conv_tri)[0]
    dS_unfold_temp_indices = conv_tri_array[non_zero_conv] - 1

    for g in range(group):
        dS_unfold_temp_start = g * max_conv

        for idx, non_zero_idx in enumerate(non_zero_conv):
            dS_unfold_ZONE[g * N_hexx + non_zero_idx] = dS_unfold_ZONE_temp[dS_unfold_temp_start + (conv_tri_array[non_zero_idx] - 1)]    

        for n in range(N_hexx):
            if conv_tri[n] == 0:
                dS_unfold_ZONE[g*N_hexx+n] = np.nan
    dS_unfold_ZONE_reshaped = np.reshape(dS_unfold_ZONE,(group,N_hexx))

    # Plot dPHI_sol_reshaped
    dS_unfold_ZONE_temp_reshaped = np.reshape(dS_unfold_ZONE_temp, (group, max_conv))
    for g in range(group):
        plot_triangular_general(dS_unfold_ZONE_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_ZONE_unfold', title=f'2D Plot of dS{g+1}_ZONE Hexx Magnitude', case_name=case_name, output_dir=output_ZONE, process_data="magnitude")
        plot_triangular_general(dS_unfold_ZONE_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_ZONE_unfold', title=f'2D Plot of dS{g+1}_ZONE Hexx Phase', case_name=case_name, output_dir=output_ZONE, process_data="phase")

    # OUTPUT
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_ZONE_groupname = f'dS_unfold{g+1}'
        dS_unfold_ZONE_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_ZONE_reshaped[g]]
        output[dS_unfold_ZONE_groupname] = dS_unfold_ZONE_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_04_ZONE/{case_name}_dS_unfold_ZONE_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    # Calculate error and compare
    diff_S1_ZONE = (np.abs(np.array(dS_unfold_ZONE_temp_reshaped[0]) - np.array(S_reshaped[0])) / (np.abs(np.array(S_reshaped[0])) + 1E-20)) * 100
    diff_S2_ZONE = (np.abs(np.array(dS_unfold_ZONE_temp_reshaped[1]) - np.array(S_reshaped[1])) / (np.abs(np.array(S_reshaped[0])) + 1E-20)) * 100
    diff_S_ZONE = [[diff_S1_ZONE], [diff_S2_ZONE]]
    diff_S_ZONE_array = np.array(diff_S_ZONE)
    diff_S_ZONE_reshaped = diff_S_ZONE_array.reshape(group, max_conv)

    for g in range(group):
        plot_triangular_general(diff_S_ZONE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_ZONE', title=f'2D Plot of diff_S{g+1}_ZONE Hexx Magnitude', case_name=case_name, output_dir=output_ZONE, process_data="magnitude")
        plot_triangular_general(diff_S_ZONE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_ZONE', title=f'2D Plot of diff_S{g+1}_ZONE Hexx Phase', case_name=case_name, output_dir=output_ZONE, process_data="phase")

##### 05. SCAN
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_05_SCAN', exist_ok=True)
    output_SCAN = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_05_SCAN/{case_name}'

    # Create tuple of detector pairs
    flux_pos = [index for index, value in enumerate(map_detector_hexx) if value == 1]

    flux_pos_conv = []
    for i, val in enumerate(flux_pos):
        flux_pos_conv.append(conv_tri[val])
    detector_pair = list(combinations(flux_pos_conv, 2))
    
    delta_all = []
    for g in range(group):
        for n in range(max_conv):
            m = g * max_conv + n
            delta_AB = 0
            for p in range(len(detector_pair)):
                det_A = detector_pair[p][0] - 1
                det_B = detector_pair[p][1] - 1

                # Retrieve values for detectors A and B
                dPHI_A = dPHI_temp_meas[g * max_conv + (det_A)]
                dPHI_B = dPHI_temp_meas[g * max_conv + (det_B)]
                G_A = G_matrix[g * max_conv + (det_A)][m]
                G_B = G_matrix[g * max_conv + (det_B)][m]

                delta_AB += np.abs((dPHI_A / dPHI_B) - (G_A / G_B))

            delta_all.append(delta_AB)
            print(f'Done for group {g+1}, position {n}')

    # Save delta_all to a text file
    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_05_SCAN/{case_name}_delta_all.txt', 'w') as f:
        for item in delta_all:
            f.write(f"{item}\n")

    # Flatten dPHI_sol_temp_groups to a 1D list
    delta_all_full = np.zeros((group* N_hexx), dtype=complex) # 1D list, size (group * N)
    non_zero_conv = np.nonzero(conv_tri)[0]

    # Create a copy for contour plotting before introducing np.nan
    for g in range(group):
        dPHI_temp_start = g * max_conv

        for idx, non_zero_idx in enumerate(non_zero_conv):
            delta_all_full[g * N_hexx + non_zero_idx] = delta_all[dPHI_temp_start + (conv_tri_array[non_zero_idx] - 1)]

    # Now assign np.nan where necessary
    for g in range(group):
        for n in range(N_hexx):
            if conv_tri[n] == 0:
                delta_all_full[g * N_hexx + n] = np.nan

    # Continue with other plotting routines
    delta_all_full_plot = np.reshape(delta_all_full, (group, N_hexx))
    delta_all_plot = np.reshape(delta_all, (group, max_conv)) #3D array, size (group, J_max, I_max)
    for g in range(group):
        plot_triangular_general(delta_all_plot[g], x, y, tri_indices, g+1, cmap='viridis', varname=f'delta_AB{g+1}', title=f'2D Plot of delta_AB{g+1} Magnitude', case_name=case_name, output_dir=output_SCAN, process_data="magnitude")

    # Flatten dPHI_sol_temp_groups to a 1D list
    delta_all_full_inv = np.zeros((group* N_hexx), dtype=complex) # 1D list, size (group * N)
    delta_all_inv = np.zeros((group* max_conv), dtype=complex) # 1D list, size (group * N)
    non_zero_conv = np.nonzero(conv_tri)[0]

    for g in range(group):
        for n in range(max_conv):
            delta_all_inv[g * max_conv + n] = 1/delta_all[g * max_conv + n]

    for g in range(group):
        dPHI_temp_start = g * max_conv

        for idx, non_zero_idx in enumerate(non_zero_conv):
            delta_all_full_inv[g * N_hexx + non_zero_idx] = delta_all_inv[dPHI_temp_start + (conv_tri_array[non_zero_idx] - 1)]    

        for n in range(N_hexx):
            if conv_tri[n] == 0:
                delta_all_full_inv[g*N_hexx+n] = np.nan

    # Plot dPHI_sol_reshaped
    delta_all_full_inv_plot = np.reshape(delta_all_full_inv, (group, N_hexx)) #3D array, size (group, J_max, I_max)
    delta_all_inv_plot = np.reshape(delta_all_inv, (group, max_conv)) #3D array, size (group, J_max, I_max)
    for g in range(group):
        plot_triangular_general(delta_all_inv_plot[g], x, y, tri_indices, g+1, cmap='viridis', varname=f'delta_AB{g+1}_inv', title=f'2D Plot of delta_AB{g+1}_inv Magnitude', case_name=case_name, output_dir=output_SCAN, process_data="magnitude")

    # Find minimum value and index
    min_value = min(delta_all)
    min_index = delta_all.index(min_value)
    for g in range(group):
        for n in range(N_hexx):
            if  g * max_conv + (conv_tri[n] - 1) == min_index:
                print(f"Minimum value is {min_value} at index {min_index} within group {g+1}")

    # Determine the scaling
    detector_loc = []
    for n in range(N_hexx):
        if map_detector_hexx[n] == 1:
            detector_loc.append(conv_tri[n]-1)

    # Determine the scaling 
    G_sol_mat_temp_new = G_matrix[detector_loc[0]][min_index]
    dPHI_temp_meas_new = dPHI_temp_meas[detector_loc[0]]

    W = dPHI_temp_meas_new/G_sol_mat_temp_new #np.abs(dPHI_temp_meas_new/G_sol_mat_temp_new)
    print(f'magnitude of dS unfold is {W}')

    dS_unfold_SCAN_temp = [0.0] * group * max_conv
    dS_unfold_SCAN_temp[min_index] = W

    # Flatten dPHI_sol_temp_groups to a 1D list
    dS_unfold_SCAN = np.zeros((group* N_hexx), dtype=complex) # 1D list, size (group * N)
    non_zero_conv = np.nonzero(conv_tri)[0]

    for g in range(group):
        dPHI_temp_start = g * max_conv

        for idx, non_zero_idx in enumerate(non_zero_conv):
            dS_unfold_SCAN[g * N_hexx + non_zero_idx] = dS_unfold_SCAN_temp[dPHI_temp_start + (conv_tri_array[non_zero_idx] - 1)]    

        for n in range(N_hexx):
            if conv_tri[n] == 0:
                dS_unfold_SCAN[g*N_hexx+n] = np.nan

    # Plot dPHI_sol_reshaped
    dS_unfold_SCAN_reshaped = np.reshape(dS_unfold_SCAN,(group,N_hexx))
    dS_unfold_SCAN_temp_reshaped = np.reshape(dS_unfold_SCAN_temp,(group,max_conv))
    for g in range(group):
        plot_triangular_general(dS_unfold_SCAN_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_SCAN_unfold', title=f'2D Plot of dS{g+1}_SCAN Hexx Magnitude', case_name=case_name, output_dir=output_SCAN, process_data="magnitude")
        plot_triangular_general(dS_unfold_SCAN_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_SCAN_unfold', title=f'2D Plot of dS{g+1}_SCAN Hexx Phase', case_name=case_name, output_dir=output_SCAN, process_data="phase")

    # OUTPUT
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_SCAN_groupname = f'dS_unfold{g+1}'
        dS_unfold_SCAN_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_SCAN_reshaped[g]]
        output[dS_unfold_SCAN_groupname] = dS_unfold_SCAN_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_05_SCAN/{case_name}_dS_SCAN_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    # Calculate error and compare
    diff_S1_SCAN = np.abs(np.array(dS_unfold_SCAN_temp_reshaped[0]) - np.array(S_reshaped[0]))/(np.abs(np.array(S_reshaped[0])) + 1E-10) * 100
    diff_S2_SCAN = np.abs(np.array(dS_unfold_SCAN_temp_reshaped[1]) - np.array(S_reshaped[1]))/(np.abs(np.array(S_reshaped[1])) + 1E-10) * 100
    diff_S_SCAN = [[diff_S1_SCAN], [diff_S2_SCAN]]
    diff_S_SCAN_array = np.array(diff_S_SCAN)
    diff_S_SCAN_reshaped = diff_S_SCAN_array.reshape(group, max_conv)

    for g in range(group):
        plot_triangular_general(diff_S_SCAN_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_SCAN', title=f'2D Plot of diff_S{g+1}_SCAN Hexx Magnitude', case_name=case_name, output_dir=output_SCAN, process_data="magnitude")
        plot_triangular_general(diff_S_SCAN_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_SCAN', title=f'2D Plot of diff_S{g+1}_SCAN Hexx Phase', case_name=case_name, output_dir=output_SCAN, process_data="phase")

##### 06. BRUTE FORCE
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_06_BRUTE', exist_ok=True)
    output_BRUTE = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_06_BRUTE/{case_name}'

    dPHI_temp_meas = dPHI_temp.copy() # 1D list, size (group * max_conv)
    for g in range(group):
        for n in range(len(map_detector_hexx)):
            if map_detector_hexx[n] == 0:
                idx = g * max_conv + (conv_tri[n]-1)
                dPHI_temp_meas[idx] = 0
    dPHI_temp_meas_reshaped = np.reshape(dPHI_temp_meas, (group, max_conv))

    # Define zeroed dPHI as dPHI_zero
    non_zero_conv = np.nonzero(conv_tri)[0]
    dPHI_temp_conv = conv_tri_array[non_zero_conv] - 1
    dPHI_meas = np.zeros((group* N_hexx), dtype=complex) # 1D list, size (group * N)

    for g in range(group):
        dPHI_temp_start = g * max(conv_tri)
        dPHI_meas[g * N_hexx + non_zero_conv] = dPHI_temp_meas[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N_hexx):
            if conv_tri[n] == 0:
                dPHI_meas[g*N_hexx+n] = np.nan

#    # Plot dPHI_zero_reshaped
#    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, J_max, I_max)) #3D array, size (group, J_max, I_max)

    for g in range(group):
        plot_triangular_general(dPHI_temp_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Magnitude', case_name=case_name, output_dir=output_BRUTE, process_data="magnitude")
        plot_triangular_general(dPHI_temp_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Phase', case_name=case_name, output_dir=output_BRUTE, process_data="phase")

    non_zero_indices = np.nonzero(dPHI_temp_meas)[0]
    dPHI_temp_meas = np.array(dPHI_temp_meas)

    # Create dictionary to store each atom (column of G_matrix_full)
    G_dictionary = {}
    for g in range(group):
        for n in range(max_conv):
            m = g * max_conv + n
            G_dictionary[f"G_g{g+1}_n{n+1}"] = G_matrix[:, m]

    # Sample the dictionary atoms at known points (sparse form)
    G_dictionary_sampled = {k: np.zeros_like(dPHI_temp_meas, dtype=complex) for k in G_dictionary}
    for k, o in G_dictionary.items():
        G_dictionary_sampled[k][non_zero_indices] = o[non_zero_indices]

    # Initialize variables
    valid_solution_BRUTE = False  # Flag to indicate a valid solution
    tol_BRUTE = 1E-8

    # Brute force over all combinations of atoms
    atom_keys = list(G_dictionary_sampled.keys())
    num_atoms = len(atom_keys)
    residual_file = f"{output_dir}/{case_name}_UNFOLDING/{case_name}_06_BRUTE/{case_name}_subset_residuals.txt"

    # Check if the file exists, if yes, delete it
    if os.path.exists(residual_file):
        os.remove(residual_file)
        print(f"Existing file '{residual_file}' deleted.")
    
    # Iterate over subsets of atoms
    for num_source in range(1, num_atoms + 1):
        print(f"Trying number of source mesh = {num_source}")
        iter_BRUTE = 0
        for subset in combinations(atom_keys, num_source):
            if iter_BRUTE % (20 * num_atoms) == 0:
                print(f"Iteration = {iter_BRUTE}, subset progress = {(iter_BRUTE/len(list(combinations(atom_keys, num_source)))*100):.2f}%, subsets = {subset}")

            # Form the initial matrix with the subset
            A = np.array([G_dictionary_sampled[k] for k in subset]).T #np.column_stack([G_dictionary_sampled[k] for k in subset])
            coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
            coefficients = dict(zip(subset, coeffs))
            residual = dPHI_temp_meas - A @ coeffs
            residual_norm = np.linalg.norm(residual)

            # Append to file without changing loop structure
            with open(residual_file, "a") as file:
                file.write(f"{subset}, {residual_norm:.6e}\n")

            # Check if residual norm meets tolerance
            if residual_norm < tol_BRUTE:
                print(f'Subsets {subset} pass the residual tolerance')
                valid_solution_BRUTE = True  # Criterion satisfied
                print(f"Valid solution found with number of sources = {num_source} and atoms = {subset}.")
                coefficients = dict(zip(subset, coeffs))
                dPHI_temp_BRUTE = sum(c * G_dictionary[k] for k, c in coefficients.items())
                break  # Exit the outer loop

            if valid_solution_BRUTE:
                break  # Exit the subset loop

            iter_BRUTE += 1

        if valid_solution_BRUTE:
            break  # Exit the outer loop
    
    if not valid_solution_BRUTE:
        print("No valid solution found with brute force.")

    ###################################################################################################
    if valid_solution_BRUTE:
        # Reshape reconstructed signal
        non_zero_conv = np.nonzero(conv_tri)[0]
        dPHI_temp_conv = conv_tri_array[non_zero_conv] - 1
        dPHI_BRUTE = np.zeros((group* N_hexx), dtype=complex) # 1D list, size (group * N)

        for g in range(group):
            dPHI_temp_start = g * max(conv_tri)
            dPHI_BRUTE[g * N_hexx + non_zero_conv] = dPHI_temp_BRUTE[dPHI_temp_start + dPHI_temp_conv]
            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    dPHI_BRUTE[g*N_hexx+n] = np.nan

        # Plot dPHI_zero_reshaped
        dPHI_temp_BRUTE_reshaped = np.reshape(dPHI_temp_BRUTE, (group, max_conv)) #3D array, size (group, J_max, I_max)
        for g in range(group):
            plot_triangular_general(dPHI_temp_BRUTE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_BRUTE', title=f'2D Plot of dPHI{g+1}_BRUTE Hexx Magnitude', case_name=case_name, output_dir=output_BRUTE, process_data="magnitude")
            plot_triangular_general(dPHI_temp_BRUTE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_BRUTE', title=f'2D Plot of dPHI{g+1}_BRUTE Hexx Phase', case_name=case_name, output_dir=output_BRUTE, process_data="phase")

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'Solve for dS using Direct Method')
        G_inverse = scipy.linalg.inv(G_matrix)

        # Plot G_matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(G_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label='Magnitude of G_inverse')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title('Plot of the Magnitude of G_inverse')
        plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_06_BRUTE/{case_name}_G_inverse.png')

        # UNFOLD ALL INTERPOLATED
        dS_unfold_BRUTE_temp = np.dot(G_inverse, dPHI_temp_BRUTE)
        dS_unfold_BRUTE_temp_reshaped = np.reshape(dS_unfold_BRUTE_temp,(group,max_conv))
        for g in range(group):
            plot_triangular_general(dS_unfold_BRUTE_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_unfold_BRUTE', title=f'2D Plot of dS{g+1}_BRUTE Hexx Magnitude', case_name=case_name, output_dir=output_BRUTE, process_data="magnitude")

        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        non_zero_conv = np.nonzero(conv_tri)[0]
        dS_unfold_temp_indices = conv_tri_array[non_zero_conv] - 1
        dS_unfold_BRUTE = np.zeros((group* N_hexx), dtype=complex)

        for g in range(group):
            dS_unfold_temp_start = g * max(conv_tri)
            dS_unfold_BRUTE[g * N_hexx + non_zero_conv] = dS_unfold_BRUTE_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    dS_unfold_BRUTE[g*N_hexx+n] = np.nan

        dS_unfold_BRUTE_reshaped = np.reshape(dS_unfold_BRUTE,(group,N_hexx))

        # OUTPUT
        print(f'Generating JSON output for dS')
        output_direct1 = {}
        for g in range(group):
            dS_unfold_direct_groupname = f'dS_unfold{g+1}'
            dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_BRUTE_reshaped[g]]
            output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_06_BRUTE/{case_name}_dS_unfold_BRUTE_output.json', 'w') as json_file:
            json.dump(output_direct1, json_file, indent=4)

        # Calculate error and compare
        diff_S1_BRUTE = np.abs(np.array(dS_unfold_BRUTE_temp_reshaped[0]) - np.array(S_reshaped[0]))/(np.abs(np.array(S_reshaped[0])) + 1E-10) * 100
        diff_S2_BRUTE = np.abs(np.array(dS_unfold_BRUTE_temp_reshaped[1]) - np.array(S_reshaped[1]))/(np.abs(np.array(S_reshaped[1])) + 1E-10) * 100
        diff_S_BRUTE = [[diff_S1_BRUTE], [diff_S2_BRUTE]]
        diff_S_BRUTE_array = np.array(diff_S_BRUTE)
        diff_S_BRUTE_reshaped = diff_S_BRUTE_array.reshape(group, max_conv)

        for g in range(group):
            plot_triangular_general(diff_S_BRUTE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_BRUTE', title=f'2D Plot of diff_S{g+1}_BRUTE Hexx Magnitude', case_name=case_name, output_dir=output_BRUTE, process_data="magnitude")
            plot_triangular_general(diff_S_BRUTE_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_BRUTE', title=f'2D Plot of diff_S{g+1}_BRUTE Hexx Phase', case_name=case_name, output_dir=output_BRUTE, process_data="phase")

###### 07. BACKWARD ELIMINATION
#    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_07_BACK', exist_ok=True)
#    output_BACK = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_07_BACK/{case_name}'
#
#    for g in range(group):
#        plot_triangular_general(dPHI_temp_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Magnitude', case_name=case_name, output_dir=output_BACK, process_data="magnitude")
#        plot_triangular_general(dPHI_temp_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Phase', case_name=case_name, output_dir=output_BACK, process_data="phase")
#
#    # Initialize variables for the higher loop
#    valid_solution_BACK = False  # Flag to indicate a valid solution
#    tol_BACK = 1E-10
#    selected_atoms = list(G_dictionary_sampled.keys())
#    residual = dPHI_meas.copy()
#    iter_BACK = 0
#    residual_norm = 1.0
#    contribution_threshold = 1e-6  # Define the contribution threshold
#    coefficients = {}
#
#    # Dictionary to store valid solutions with term counts
#    valid_solutions_BACK = {}
#
#    #while not valid_solution:
#    while selected_atoms:
#        iter_BACK += 1
#        print(f"Iteration {iter_BACK}: Atoms remaining = {len(selected_atoms)}")
#
#        try:
#            # Stack all selected atoms into the matrix
#            A = np.array([G_dictionary_sampled[k] for k in selected_atoms]).T
#            coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
#            coefficients = dict(zip(selected_atoms, coeffs))
#            residual = dPHI_temp_meas - A @ coeffs
#            residual_norm = np.linalg.norm(residual)
#            # Compute contributions (absolute value of coefficients)
#            contributions = {atom: abs(coeff) / max(abs(coeffs)) for atom, coeff in zip(selected_atoms, coeffs)}
#
#            # Validate the reconstructed signal against the criterion
#            if residual_norm < tol:
#                valid_solution = True  # Criterion satisfied
#                print(f"Valid solution found, selected atoms = {selected_atoms}, residual norm = {residual_norm:.6e}")
#                valid_solutions_BACK[iter] = selected_atoms[:]
#                atom_to_remove = min(contributions, key=contributions.get)
#                selected_atoms.remove(atom_to_remove)
#            else:
#                # Find the least contributing atom
#                atom_to_remove = min(contributions, key=contributions.get)
#                selected_atoms.remove(atom_to_remove)
#                print(f"Criteria not met. Removing least contributing atom: {atom_to_remove}, Contribution = {contributions[atom_to_remove]:.6e}, residual norm = {residual_norm:.6e}")
#                if len(selected_atoms) == 0:
#                    print(f'Criteria not met using Backward Elimination.')
#                    break
#
#        except np.linalg.LinAlgError:
#            print("SVD did not converge, skipping this iteration.")
#            sorted_contributions = sorted(contributions.items(), key=lambda x: x[1])
#            second_least_atom = sorted_contributions[1][0]  # Get the atom with the second smallest contribution
#            selected_atoms.remove(second_least_atom)
#            continue  # Skip to the next iteration
#
#    if valid_solutions_BACK:
#        best_atom = min(valid_solutions_BACK, key=lambda k: len(valid_solutions_BACK[k]))
#        print(f"The best valid solution is with atom {best_atom} with iteration number = {valid_solutions_BACK[best_atom]}.")
#        valid_solution_BACK = valid_solutions_BACK[best_atom]
#
#        A = np.array([G_dictionary_sampled[k] for k in valid_solution_BACK]).T
#        coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
#        coefficients = dict(zip(valid_solution_BACK, coeffs))
#        dPHI_temp_BACK = sum(c * G_dictionary[k] for k, c in coefficients.items())
#    else:
#        print("Failed to find a valid solution within the maximum number of outer iterations.")
#
#    ###################################################################################################
#    if valid_solution_BACK:
#        # Reshape reconstructed signal
#        non_zero_conv = np.nonzero(conv_tri)[0]
#        dPHI_temp_conv = conv_tri_array[non_zero_conv] - 1
#        dPHI_BACK = np.zeros((group* N_hexx), dtype=complex) # 1D list, size (group * N)
#
#        for g in range(group):
#            dPHI_temp_start = g * max(conv_tri)
#            dPHI_BACK[g * N_hexx + non_zero_conv] = dPHI_temp_BACK[dPHI_temp_start + dPHI_temp_conv]
#            for n in range(N_hexx):
#                if conv_tri[n] == 0:
#                    dPHI_BACK[g*N_hexx+n] = np.nan
#
#        # Plot dPHI_zero_reshaped
#        dPHI_BACK_reshaped = np.reshape(dPHI_temp_BACK, (group, max_conv))
#        for g in range(group):
#            plot_triangular_general(dPHI_BACK_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_BACK', title=f'2D Plot of dPHI{g+1}_BACK Hexx Magnitude', case_name=case_name, output_dir=output_BACK, process_data="magnitude")
#            plot_triangular_general(dPHI_BACK_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_BACK', title=f'2D Plot of dPHI{g+1}_BACK Hexx Phase', case_name=case_name, output_dir=output_BACK, process_data="phase")
#
#        ######################################################################################################
#        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
#        print(f'Solve for dS using Direct Method')
#        G_inverse = scipy.linalg.inv(G_matrix)
#
#        # Plot G_matrix
#        plt.figure(figsize=(8, 6))
#        plt.imshow(G_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
#        plt.colorbar(label='Magnitude of G_inverse')
#        plt.xlabel('Index')
#        plt.ylabel('Index')
#        plt.gca().invert_yaxis()
#        plt.title('Plot of the Magnitude of G_inverse')
#        plt.savefig(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_07_BACK/{case_name}_G_inverse.png')
#
#        # UNFOLD ALL INTERPOLATED
#        dS_unfold_BACK_temp = np.dot(G_inverse, dPHI_temp_BACK)
#        dS_unfold_BACK_temp_reshaped = np.reshape(dS_unfold_BACK_temp,(group,max_conv))
#        for g in range(group):
#            plot_triangular_general(dS_unfold_BACK_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_unfold_BACK', title=f'2D Plot of dS{g+1}_BACK Hexx Magnitude', case_name=case_name, output_dir=output_BACK, process_data="magnitude")
#
#        # POSTPROCESS
#        print(f'Postprocessing to appropriate dPHI')
#        non_zero_conv = np.nonzero(conv_tri)[0]
#        dS_unfold_temp_indices = conv_tri_array[non_zero_conv] - 1
#        dS_unfold_BACK = np.zeros((group* N_hexx), dtype=complex)
#
#        for g in range(group):
#            dS_unfold_temp_start = g * max(conv_tri)
#            dS_unfold_BACK[g * N_hexx + non_zero_conv] = dS_unfold_BACK_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
#            for n in range(N_hexx):
#                if conv_tri[n] == 0:
#                    dS_unfold_BACK[g*N_hexx+n] = np.nan
#
#        dS_unfold_BACK_reshaped = np.reshape(dS_unfold_BACK,(group,N_hexx))
#
#        # OUTPUT
#        print(f'Generating JSON output for dS')
#        output_direct1 = {}
#        for g in range(group):
#            dS_unfold_direct_groupname = f'dS_unfold{g+1}'
#            dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_BACK_reshaped[g]]
#            output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list
#
#        # Save data to JSON file
#        with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_07_BACK/{case_name}_dS_unfold_BACK_output.json', 'w') as json_file:
#            json.dump(output_direct1, json_file, indent=4)
#
#        # Calculate error and compare
#        diff_S1_BACK = np.abs(np.array(dS_unfold_BACK_temp_reshaped[0]) - np.array(S_reshaped[0]))/(np.abs(np.array(S_reshaped[0])) + 1E-7) * 100
#        diff_S2_BACK = np.abs(np.array(dS_unfold_BACK_temp_reshaped[1]) - np.array(S_reshaped[1]))/(np.abs(np.array(S_reshaped[1])) + 1E-7) * 100
#        diff_S_BACK = [[diff_S1_BACK], [diff_S2_BACK]]
#        diff_S_BACK_array = np.array(diff_S_BACK)
#        diff_S_BACK_reshaped = diff_S_BACK_array.reshape(group, max_conv)
#
#        for g in range(group):
#            plot_triangular_general(diff_S_BACK_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_BACK', title=f'2D Plot of diff_S{g+1}_BACK Hexx Magnitude', case_name=case_name, output_dir=output_BACK, process_data="magnitude")
#            plot_triangular_general(diff_S_BACK_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_BACK', title=f'2D Plot of diff_S{g+1}_BACK Hexx Phase', case_name=case_name, output_dir=output_BACK, process_data="phase")
#
#    else:
#        print("No valid solution found with backward elimination.")
#
##### 08. GREEDY
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_08_GREEDY', exist_ok=True)
    output_GREEDY = f'{output_dir}/{case_name}_UNFOLDING/{case_name}_08_GREEDY/{case_name}'

    for g in range(group):
        plot_triangular_general(dPHI_temp_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Magnitude', case_name=case_name, output_dir=output_GREEDY, process_data="magnitude")
        plot_triangular_general(dPHI_temp_meas_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_meas', title=f'2D Plot of dPHI{g+1}_meas Hexx Phase', case_name=case_name, output_dir=output_GREEDY, process_data="phase")

    # Initialize variables for the higher loop
    valid_solution_GREEDY = False  # Flag to indicate a valid solution
    outer_iter = 0
    inner_iter = 0
    tol_GREEDY = 1E-10  # Stopping tolerance
    comb_first_atom = 1
    selected_atoms = []
    contribution_threshold = 1e-6  # Define the contribution threshold
    all_outer_iter_len = len(G_dictionary) #+ len(list(combinations(G_dictionary_sampled.keys(), 2)))

    # Dictionary to store valid solutions with term counts
    valid_solutions_GREEDY = {}

    # Iterate over possible first atoms
    while outer_iter < all_outer_iter_len:
        first_atom_iter = combinations(G_dictionary_sampled.keys(), comb_first_atom)

        for first_atom in first_atom_iter:
            # Initialize residual and coefficients
            residual = dPHI_temp_meas.copy()
            selected_atoms = list(first_atom)
            coefficients = []
            residual_norm = np.linalg.norm(residual)

            # Form the initial matrix with the first atom
            A = np.array([G_dictionary_sampled[k] for k in first_atom]).T
            coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
            residual = dPHI_temp_meas - A @ coeffs
            residual_norm = np.linalg.norm(residual)
            print(f"Outer iteration {outer_iter+1}: Trying first atom {first_atom}, current residual norm = {residual_norm:.6e}")

            # Perform Greedy Residual Minimization
            prev_selected_atoms_len = 0
            constant_len_counter = 0
            while residual_norm > tol_GREEDY:
                residuals = {}
                for k in combinations(G_dictionary_sampled.keys(), comb_first_atom):
                    # Skip this combination if any atom is already selected
                    if any(atom in selected_atoms for atom in k):
                        continue
                    temp_atoms = selected_atoms + list(k) #[k]
                    A_temp = np.array([G_dictionary_sampled[a] for a in temp_atoms]).T
                    coeffs_temp = np.linalg.lstsq(A_temp, dPHI_temp_meas, rcond=None)[0]
                    residuals[k] = np.linalg.norm(dPHI_temp_meas - A_temp @ coeffs_temp)
                chosen_atom = min(residuals, key=residuals.get)

                if isinstance(chosen_atom, tuple):  # If chosen_atom is a tuple, extend the list
                    for atom in chosen_atom:
                        if atom not in selected_atoms:
                            selected_atoms.append(atom)
                else:  # If chosen_atom is a single key, append it
                    if chosen_atom not in selected_atoms:
                        selected_atoms.append(chosen_atom)

                # Form matrix of selected atoms
                A = np.array([G_dictionary_sampled[k] for k in selected_atoms]).T

                # Solve least-squares problem to update coefficients
                coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
                coefficients = dict(zip(selected_atoms, coeffs))

                # Update residual
                residual = dPHI_temp_meas - A @ coeffs
                residual_norm = np.linalg.norm(residual)

                print(f'   Chosen atom = {chosen_atom}, length of selected atoms = {len(selected_atoms)}, current residual norm = {residual_norm:.6e}')

                # Check if the length of selected_atoms remains constant
                if len(selected_atoms) == prev_selected_atoms_len:
                    constant_len_counter += 1
                else:
                    constant_len_counter = 0  # Reset counter if length changes

                prev_selected_atoms_len = len(selected_atoms)

                if constant_len_counter >= 10:
                    print("   Terminating loop: Length of selected_atoms remained constant for 10 iterations.")
                    break

                inner_iter += 1

            # Check for low contribution atoms
            contributions = {atom: abs(coeff) / max(abs(coeffs)) for atom, coeff in zip(selected_atoms, coeffs)}
            low_contribution_atoms = [atom for atom, contribution in contributions.items() if contribution < contribution_threshold]

            if low_contribution_atoms:
                for atom in low_contribution_atoms:
                    if atom in selected_atoms:
                        selected_atoms.remove(atom)
            print(f"   Selected_atoms = {selected_atoms}, residual norm = {residual_norm:.6e}")

            # Validate the reconstructed signal against the criterion
            if residual_norm < tol_GREEDY:
                valid_solution = True  # Criterion satisfied
                print(f"Valid solution found with first atom {first_atom} in outer iteration {outer_iter+1}.")
                valid_solutions_GREEDY[first_atom] = selected_atoms #len(selected_atoms)
            else:
                print(f"Criterion not met with first atom {first_atom}. Restarting with a new atom.")

            outer_iter += 1

    #    prev_comb_first_atom = comb_first_atom
    #    comb_first_atom += 1

    # Final check for the best solution
    if valid_solutions_GREEDY:
        best_atom = min(valid_solutions_GREEDY, key=lambda k: len(valid_solutions_GREEDY[k]))
        print(f"The best valid solution is with atom {best_atom} with selected atoms = {valid_solutions_GREEDY[best_atom]}.")
        valid_solution_GREEDY = valid_solutions_GREEDY[best_atom]

        A = np.array([G_dictionary_sampled[k] for k in valid_solution_GREEDY]).T
        coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
        coefficients = dict(zip(valid_solution_GREEDY, coeffs))
        dPHI_temp_GREEDY = sum(c * G_dictionary[k] for k, c in zip(valid_solution_GREEDY, coeffs))
    else:
        print("Failed to find a valid solution within the maximum number of outer iterations.")

    ####################################################################################################
    if valid_solution_GREEDY:
        # Reshape reconstructed signal
        non_zero_conv = np.nonzero(conv_tri)[0]
        dPHI_temp_conv = conv_tri_array[non_zero_conv] - 1
        dPHI_GREEDY = np.zeros((group* N_hexx), dtype=complex) # 1D list, size (group * N)

        for g in range(group):
            dPHI_temp_start = g * max(conv_tri)
            dPHI_GREEDY[g * N_hexx + non_zero_conv] = dPHI_temp_GREEDY[dPHI_temp_start + dPHI_temp_conv]
            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    dPHI_GREEDY[g*N_hexx+n] = np.nan

        # Plot dPHI_zero_reshaped
        dPHI_temp_GREEDY_reshaped = np.reshape(dPHI_temp_GREEDY, (group, max_conv))
        for g in range(group):
            plot_triangular_general(dPHI_temp_GREEDY_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_GREEDY', title=f'2D Plot of dPHI{g+1}_GREEDY Hexx Magnitude', case_name=case_name, output_dir=output_GREEDY, process_data="magnitude")
            plot_triangular_general(dPHI_temp_GREEDY_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_GREEDY', title=f'2D Plot of dPHI{g+1}_GREEDY Hexx Phase', case_name=case_name, output_dir=output_GREEDY, process_data="phase")

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'\nSolve for dS using Direct Method')

        # UNFOLD ALL INTERPOLATED
        dS_unfold_GREEDY_temp = np.dot(G_inverse, dPHI_temp_GREEDY)
    
        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        non_zero_conv = np.nonzero(conv_tri)[0]
        dS_unfold_temp_indices = conv_tri_array[non_zero_conv] - 1
        dS_unfold_GREEDY = np.zeros((group* N_hexx), dtype=complex)

        for g in range(group):
            dS_unfold_temp_start = g * max(conv_tri)
            dS_unfold_GREEDY[g * N_hexx + non_zero_conv] = dS_unfold_GREEDY_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    dS_unfold_GREEDY[g*N_hexx+n] = np.nan

        dS_unfold_GREEDY_reshaped = np.reshape(dS_unfold_GREEDY,(group, N_hexx))
        dS_unfold_GREEDY_temp_reshaped = np.reshape(dS_unfold_GREEDY_temp,(group, max_conv))

        for g in range(group):
            plot_triangular_general(dS_unfold_GREEDY_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_unfold_GREEDY', title=f'2D Plot of dS{g+1}_unfold_GREEDY Hexx Magnitude', case_name=case_name, output_dir=output_GREEDY, process_data="magnitude")
            plot_triangular_general(dS_unfold_GREEDY_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dS_unfold_GREEDY', title=f'2D Plot of dS{g+1}_unfold_GREEDY Hexx Phase', case_name=case_name, output_dir=output_GREEDY, process_data="phase")

        # OUTPUT
        print(f'Generating JSON output for dS')
        output_direct1 = {}
        for g in range(group):
            dS_unfold_direct_groupname = f'dS_unfold{g+1}'
            dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_GREEDY_reshaped[g]]
            output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_08_GREEDY/{case_name}_dS_unfold_GREEDY_output.json', 'w') as json_file:
            json.dump(output_direct1, json_file, indent=4)

        # Calculate error and compare
        diff_S1_GREEDY = np.abs(np.array(dS_unfold_GREEDY_temp_reshaped[0]) - np.array(S_reshaped[0]))/(np.abs(np.array(S_reshaped[0])) + 1E-6) * 100
        diff_S2_GREEDY = np.abs(np.array(dS_unfold_GREEDY_temp_reshaped[1]) - np.array(S_reshaped[1]))/(np.abs(np.array(S_reshaped[1])) + 1E-6) * 100
        diff_S_GREEDY = [[diff_S1_GREEDY], [diff_S2_GREEDY]]
        diff_S_GREEDY_array = np.array(diff_S_GREEDY)
        diff_S_GREEDY_reshaped = diff_S_GREEDY_array.reshape(group,max_conv)

        for g in range(group):
            plot_triangular_general(diff_S_GREEDY_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_GREEDY', title=f'2D Plot of diff_S{g+1}_GREEDY Hexx Magnitude', case_name=case_name, output_dir=output_GREEDY, process_data="magnitude")
            plot_triangular_general(diff_S_GREEDY_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_S_GREEDY', title=f'2D Plot of diff_S{g+1}_GREEDY Hexx Phase', case_name=case_name, output_dir=output_GREEDY, process_data="phase")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed: {elapsed_time:3e} seconds')

if __name__ == "__main__":
    main()