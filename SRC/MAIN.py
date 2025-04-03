import numpy as np
import json
import time
import os
import sys

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

#from INPUTS.TASK1_TEST01_1D1G_general import *
#from INPUTS.TASK1_TEST02_1DMG_CSTest03 import *
#from INPUTS.TASK1_TEST03_1DMG_CSTest05 import *
#from INPUTS.TASK1_TEST04_1DMG_CSTest07 import * # No noise case

#from INPUTS.TASK1_TEST05_2DMG_Serp289_2g import * # No noise case
#from INPUTS.TASK1_TEST06_2DMG_CSTest10_VandV import *
#from INPUTS.TASK1_TEST07_2DMG_C3_VandV import *
#from INPUTS.TASK1_TEST08_2DMG_BIBLIS_VandV import * # No noise case
#from INPUTS.TASK1_TEST09_2DMG_PWRMOX_VandV import *

#from INPUTS.TASK1_TEST15_2DTriMG_3ring import * # No noise case
from INPUTS.TASK1_TEST16_2DTriMG_HOMOG_VandV import *
#from INPUTS.TASK1_TEST17_2DTriMG_VVER400_VandV import * # No noise case
#from INPUTS.TASK1_TEST18_2DTriMG_HTTR2G_VandV import *
#from INPUTS.TASK1_TEST19_2DTriMG_HTTR4G import * # No noise case
#from INPUTS.TASK1_TEST20_2DTriMG_HTTR7G import * # No noise case
#from INPUTS.TASK1_TEST21_2DTriMG_HTTR14G import * # No noise case

# Restore the original sys.path
sys.path = original_sys_path

#######################################################################################################
solver_type = 'forward'
#solver_type = 'adjoint'
#solver_type = 'noise'

#######################################################################################################
def main():
    start_time = time.time()

    if geom_type =='1D':
        output_dir = f'../OUTPUTS/{case_name}'
        x = globals().get("x")
        dx = globals().get("dx")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS = globals().get("SIGS")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")

        Utils.create_directories(solver_type, output_dir, case_name)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F, dx, precond, tol=1E-10)
            keff, PHI = solver.solve()
            PHI_reshaped = np.reshape(PHI, (group, N))
            PostProcessor.save_output_power1D(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_1D_power(solver_type, PHI_reshaped[g], x, g, output_dir=output_dir, varname=f'PHI', case_name=case_name, title=f'1D Plot of PHI{g+1}')
        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS = globals().get("dSIGS")
            dNUFIS = globals().get("dNUFIS")
            dSOURCE = globals().get("dSOURCE")
            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
            keff = forward_output["keff"]
            PHI = []
            for i in range(group):
                phi_key = f"PHI{i+1}_FORWARD"
                PHI.extend(forward_output[phi_key])

            matrix_builder = MatrixBuilderNoise1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS, dNUFIS)
            M, dS = matrix_builder.build_noise_matrices()

            solver = SolverFactory.get_solver_fixed1D(solver_type, group, N, M, dS, dSOURCE, PHI, dx, precond, tol=1e-10)

            dPHI = solver.solve()
            dPHI_reshaped = np.reshape(dPHI, (group, N))
            PostProcessor.save_output_fixed1D(output_dir, case_name, dPHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_1D_fixed(solver_type, dPHI_reshaped[g], x, g, output_dir=output_dir, varname=f'dPHI', case_name=case_name, title=f'1D Plot of dPHI{g+1}')

    elif geom_type =='2D rectangular':
        x = globals().get("x")
        y = globals().get("y")
        dx = globals().get("dx")
        dy = globals().get("dy")
        I_max = globals().get("I_max")
        J_max = globals().get("J_max")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS_reshaped = globals().get("SIGS_reshaped")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")

        output_dir = f'../OUTPUTS/{case_name}'
        Utils.create_directories(solver_type, output_dir, case_name)
        conv = convert_index_2D_rect(D, I_max, J_max)
        conv_array = np.array(conv)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power2DRect(solver_type, group, N, conv, M, F, dx, dy, precond, tol=1E-10)
            keff, phi_temp = solver.solve()

            PHI, PHI_reshaped, PHI_reshaped_plot = PostProcessor.postprocess_power2DRect(phi_temp, conv, group, N, I_max, J_max)
            PostProcessor.save_output_power2DRect(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_2D_rect_power(solver_type, PHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'PHI', case_name=case_name, title=f'2D Plot of PHI{g+1}')
        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS_reshaped = globals().get("dSIGS_reshaped")
            dNUFIS = globals().get("dNUFIS")

            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
            keff = forward_output["keff"]
            PHI_all = []
            for i in range(group):
                phi_key = f"PHI{i+1}_FORWARD"
                PHI_all.append(forward_output[phi_key])

            PHI = np.zeros(max(conv) * group)
            for g in range(group):
                PHI_indices = g * max(conv) + (conv_array - 1)
                PHI[PHI_indices] = PHI_all[g]

            matrix_builder = MatrixBuilderNoise2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS)
            M, dS = matrix_builder.build_noise_matrices()

            solver = SolverFactory.get_solver_fixed2DRect(solver_type, group, N, conv, M, dS, PHI, dx, dy, precond, tol=1e-10)

            dPHI_temp = solver.solve()
            dPHI, dPHI_reshaped, dPHI_reshaped_plot = PostProcessor.postprocess_fixed2DRect(dPHI_temp, conv, group, N, I_max, J_max)
            PostProcessor.save_output_fixed2DRect(output_dir, case_name, keff, dPHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_2D_rect_fixed(solver_type, dPHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'dPHI', case_name=case_name, title=f'2D Plot of dPHI{g+1} Magnitude', process_data='magnitude')
                Utils.plot_2D_rect_fixed(solver_type, dPHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'dPHI', case_name=case_name, title=f'2D Plot of dPHI{g+1} Phase', process_data='phase')

    elif geom_type =='2D triangular':
        h = globals().get("h")
        s = globals().get("s")
        N_hexx = globals().get("N_hexx")
        level = globals().get("level")
        I_max = globals().get("I_max")
        J_max = globals().get("J_max")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS_reshaped = globals().get("SIGS_reshaped")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")
        input_name = globals().get("input_name")

        output_dir = f'../OUTPUTS/{input_name}'
        Utils.create_directories(solver_type, output_dir, case_name)
        conv_hexx = convert_2D_hexx(I_max, J_max, D)
        conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
        conv_tri_array = np.array(conv_tri)
        conv_neighbor, tri_indices, x, y, all_triangles = calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F, h, precond, tol=1E-10)
            keff, phi_temp = solver.solve()

            PHI, PHI_reshaped, PHI_temp_reshaped = PostProcessor.postprocess_power2DHexx(phi_temp, conv_tri, group, N_hexx)
            PostProcessor.save_output_power2DHexx(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                plot_triangular(PHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS_reshaped = globals().get("dSIGS_reshaped")
            dNUFIS = globals().get("dNUFIS")
            noise_section = globals().get("noise_section")
            type_noise = globals().get("type_noise")

            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
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

#            if type_noise == 'FVX' or type_noise == 'FAV':
#                if level != 4:
#                    print('Vibrating Assembly type noise only works if level = 4. Changing level to 4')
#                    level = 4

            hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)
            triangle_neighbors_global = find_triangle_neighbors_2D(all_triangles, precision=6)

            matrix_builder = MatrixBuilderNoise2DHexx(group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise)
            M, dS = matrix_builder.build_noise_matrices()

            solver = SolverFactory.get_solver_fixed2DHexx(solver_type, group, conv_tri, M, dS, PHI, precond, tol=1e-10)

            dPHI_temp = solver.solve()
            dPHI, dPHI_reshaped, dPHI_temp_reshaped = PostProcessor.postprocess_fixed2DHexx(dPHI_temp, conv_tri, group, N_hexx)
            PostProcessor.save_output_fixed2DHexx(output_dir, case_name, dPHI_reshaped, solver_type)
            for g in range(group):
                plot_triangular(dPHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
                plot_triangular(dPHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx Phase', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="phase")

    elapsed_time = time.time() - start_time
    print(f'Time elapsed: {elapsed_time:.3e} seconds')

    # SOME INFORMATION
    info_output = f"""
    --- Simulation Summary ---
    Case Name: {case_name}
    Simulation Type: {solver_type} Simulation
    Number of groups: {group}
    Dimensions: {geom_type}
    Final keff: {keff:.6f}
    Elapsed Time: {elapsed_time:.3e} seconds
    Solver Used: {'ILU' if precond == 1 else 'LU' if precond == 2 else 'Sparse Direct Solver'}
    """

    # Save the summary to a text file
    summary_file = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_summary.txt'
    with open(summary_file, 'w') as file:
        file.write(info_output)

if __name__ == "__main__":
    main()
