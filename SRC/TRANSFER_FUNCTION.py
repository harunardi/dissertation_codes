import numpy as np
import json
import time
import os
import sys
from scipy.integrate import trapezoid

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

#from INPUTS.TASK1_TEST02_1DMG_CSTest03 import *
#from INPUTS.TASK1_TEST03_1DMG_CSTest05 import *

#from INPUTS.TASK1_TEST07_2DMG_C3_VandV import *
#from INPUTS.TASK1_TEST09_2DMG_PWRMOX_VandV import *

from INPUTS.TASK1_TEST18_2DTriMG_HTTR2G_VandV import *

# Restore the original sys.path
sys.path = original_sys_path

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

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)
        matrix_builder = MatrixBuilderForward1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
        M, F_FORWARD = matrix_builder.build_forward_matrices()
        solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F_FORWARD, dx, precond, tol=1E-6)
        keff, PHI = solver.solve()
        PHI_reshaped = np.reshape(PHI, (group, N))

        output = {"keff": keff}
        for g in range(len(PHI_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)
        matrix_builder = MatrixBuilderAdjoint1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
        M, F_ADJOINT = matrix_builder.build_adjoint_matrices()
        solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F_ADJOINT, dx, precond, tol=1E-6)
        keff, PHI_ADJ = solver.solve()
        PHI_ADJ_reshaped = np.reshape(PHI_ADJ, (group, N))

        output = {"keff": keff}
        for g in range(len(PHI_ADJ_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS = globals().get("dSIGS")
        dNUFIS = globals().get("dNUFIS")
        dSOURCE = globals().get("dSOURCE")
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)
        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
            forward_output = json.load(json_file)
        keff = forward_output["keff"]
        PHI = []
        for i in range(group):
            phi_key = f"PHI{i+1}_FORWARD"
            PHI.extend(forward_output[phi_key])

        dPOWER = []
        dRHO = []

        freq = np.logspace(-4, 4, num=101)

        for f in range(len(freq)):
            ff = freq[f]
            print(f'Solving for frequency {ff:.3e}')
            omega = 2 * np.pi * ff

            matrix_builder = MatrixBuilderNoise1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS, dNUFIS)
            M, dS = matrix_builder.build_noise_matrices()

            solver = SolverFactory.get_solver_fixed1D(solver_type, group, N, M, dS, dSOURCE, PHI, dx, precond, tol=1e-06)

            dPHI = solver.solve()
            dPHI_reshaped = np.reshape(dPHI, (group, N))
            output = {}
            for g in range(len(dPHI_reshaped)):
                dPHI_groupname = f'dPHI{g + 1}'
                dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
                output[dPHI_groupname] = dPHI_list
    
            with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_freq{f}_output.json', 'w') as json_file:
                json.dump(output, json_file, indent=4)

            NUMER = 0
            DENOM = 0
            for g in range(group):
                NUMER += trapezoid((1/v[g]) * (PHI_ADJ_reshaped[g] * dPHI_reshaped[g]), dx=dx, axis = 0)
                DENOM += trapezoid((1/v[g]) * (PHI_ADJ_reshaped[g] * PHI_reshaped[g]), dx=dx, axis = 0)

            dPOWER.append(NUMER / DENOM)

            S_NUMER = dS.dot(PHI)
            S_NUMER_reshaped = np.reshape(S_NUMER, (group, N))
            S_DENOM = F_FORWARD.dot(PHI)
            S_DENOM_reshaped = np.reshape(S_DENOM, (group, N))

            NUMER = 0
            DENOM = 0
            for g in range(group):
                NUMER += trapezoid(PHI_ADJ_reshaped[g] * S_NUMER_reshaped[g], dx=dx, axis = 0)
                DENOM += trapezoid(PHI_ADJ_reshaped[g] * S_DENOM_reshaped[g], dx=dx, axis = 0)

            dRHO.append(NUMER / DENOM)

        G_0 = []
        mag_G_0 = []
        phase_G_0 = []
        for f in range(len(freq)):
            transfer_function = dPOWER[f]/dRHO[f]
            G_0.append(transfer_function)
            mag_G_0.append(abs(transfer_function))
            phase_G_0.append(np.degrees(np.angle(transfer_function)))

        # OUTPUT
        print(f'Generating JSON output')
        output = {}
        G_0_groupname = f'G_0'
        G_0_list = [{"real": x.real, "imaginary": x.imag} for x in G_0]
        output[G_0_groupname] = G_0_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_G_0_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Plotting magnitude
        plt.figure(figsize=(8, 6))  # Create a new figure
        plt.plot(freq, mag_G_0, marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude of Neutron Noise')
        plt.title('Plot of Transfer Function (Magnitude)')
        plt.savefig(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_Magnitude.png')

        # Plotting phase
        plt.figure(figsize=(8, 6))  # Create another new figure
        plt.plot(freq, phase_G_0, marker='o')
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylim(-120, 0)
        plt.ylabel('Phase of Neutron Noise')
        plt.title('Plot of Transfer Function (Phase)')
        plt.savefig(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_Phase.png')

    elif geom_type =='2D rectangular':
        output_dir = f'../OUTPUTS/{case_name}'
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

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv = convert_index_2D_rect(D, I_max, J_max)
        conv_array = np.array(conv)

        matrix_builder = MatrixBuilderForward2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
        M, F_FORWARD = matrix_builder.build_forward_matrices()

        solver = SolverFactory.get_solver_power2DRect(solver_type, group, N, conv, M, F_FORWARD, dx, dy, precond, tol=1E-10)
        keff, PHI_temp = solver.solve()
        PHI, PHI_reshaped, PHI_reshaped_plot = PostProcessor.postprocess_power2DRect(PHI_temp, conv, group, N, I_max, J_max)

        output = {"keff": keff.real}
        for g in range(len(PHI_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        PHI_temp_reshaped = np.reshape(PHI_temp, (group, max(conv)))

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv = convert_index_2D_rect(D, I_max, J_max)
        conv_array = np.array(conv)

        matrix_builder = MatrixBuilderAdjoint2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
        M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

        solver = SolverFactory.get_solver_power2DRect(solver_type, group, N, conv, M, F_ADJOINT, dx, dy, precond, tol=1E-10)
        keff, PHI_ADJ_temp = solver.solve()
        PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_reshaped_plot = PostProcessor.postprocess_power2DRect(PHI_ADJ_temp, conv, group, N, I_max, J_max)

        output = {"keff": keff.real}
        for g in range(len(PHI_ADJ_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        PHI_ADJ_temp_reshaped = np.reshape(PHI_ADJ_temp, (group, max(conv)))

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS_reshaped = globals().get("dSIGS_reshaped")
        dNUFIS = globals().get("dNUFIS")
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv = convert_index_2D_rect(D, I_max, J_max)
        conv_array = np.array(conv)

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
            forward_output = json.load(json_file)
        keff = forward_output["keff"]
        PHI_all = []
        for i in range(group):
            phi_key = f"PHI{i+1}_FORWARD"
            PHI_all.append(forward_output[phi_key])

        PHI = np.zeros(max(conv) * group)
        for g in range(group):
            PHI_indices = g * max(conv) + (conv_array - 1)
            PHI[PHI_indices] = PHI_all[g]

        dPOWER = []
        dRHO = []

        freq = np.logspace(-4, 4, num=101)

        for f in range(len(freq)):
            ff = freq[f]
            print(f'Solving for frequency {ff:.3e}')
            omega = 2 * np.pi * ff

            matrix_builder = MatrixBuilderNoise2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS)
            M, dS = matrix_builder.build_noise_matrices()

            solver = SolverFactory.get_solver_fixed2DRect(solver_type, group, N, conv, M, dS, PHI, dx, dy, precond, tol=1e-10)

            dPHI_temp = solver.solve()
            dPHI, dPHI_reshaped, dPHI_reshaped_plot = PostProcessor.postprocess_fixed2DRect(dPHI_temp, conv, group, N, I_max, J_max)
            output = {}
            for g in range(len(dPHI_reshaped)):
                dPHI_groupname = f'dPHI{g + 1}'
                dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
                output[dPHI_groupname] = dPHI_list
    
            with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_freq{f}_output.json', 'w') as json_file:
                json.dump(output, json_file, indent=4)

            dPHI_temp_reshaped = np.reshape(dPHI_temp, (group, max(conv)))

            v1_PP = v1[0][0]
            v2_PP = v2[0][0]
            v_new = [v1_PP, v2_PP]

            NUMER = 0
            DENOM = 0
            for g in range(group):
                NUMER += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * dPHI_temp_reshaped[g]), dx=dx, axis = 0)
                DENOM += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * PHI_temp_reshaped[g]), dx=dx, axis = 0)

            dPOWER.append(NUMER / DENOM)

            S_NUMER = dS.dot(PHI)
            S_NUMER_reshaped = np.reshape(S_NUMER, (group, max(conv)))
            S_DENOM = F_FORWARD.dot(PHI)
            S_DENOM_reshaped = np.reshape(S_DENOM, (group, max(conv)))
            S_NUMER_reshaped = np.nan_to_num(S_NUMER_reshaped, nan=0)
            S_DENOM_reshaped = np.nan_to_num(S_DENOM_reshaped, nan=0)

            NUMER = 0
            DENOM = 0
            for g in range(group):
                NUMER += trapezoid(PHI_ADJ_temp_reshaped[g] * S_NUMER_reshaped[g], dx=dx, axis = 0)
                DENOM += trapezoid(PHI_ADJ_temp_reshaped[g] * S_DENOM_reshaped[g], dx=dx, axis = 0)

            dRHO.append(NUMER / DENOM)

        G_0 = []
        mag_G_0 = []
        phase_G_0 = []
        for f in range(len(freq)):
            transfer_function = dPOWER[f]/dRHO[f]
            G_0.append(transfer_function)
            mag_G_0.append(abs(transfer_function))
            phase_G_0.append(np.degrees(np.angle(transfer_function)))

        # OUTPUT
        print(f'Generating JSON output')
        output = {}
        G_0_groupname = f'G_0'
        G_0_list = [{"real": x.real, "imaginary": x.imag} for x in G_0]
        output[G_0_groupname] = G_0_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_G_0_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Plotting magnitude
        plt.figure(figsize=(8, 6))  # Create a new figure
        plt.plot(freq, mag_G_0, marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude of Neutron Noise')
        plt.title('Plot of Transfer Function (Magnitude)')
        plt.savefig(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_Magnitude.png')

        # Plotting phase
        plt.figure(figsize=(8, 6))  # Create another new figure
        plt.plot(freq, phase_G_0, marker='o')
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylim(-120, 0)
        plt.ylabel('Phase of Neutron Noise')
        plt.title('Plot of Transfer Function (Phase)')
        plt.savefig(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_Phase.png')

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

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv_hexx = convert_2D_hexx(I_max, J_max, D)
        conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
        conv_tri_array = np.array(conv_tri)
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

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)

        matrix_builder = MatrixBuilderAdjoint2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
        M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

        solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F_ADJOINT, h, precond, tol=1E-10)
        keff, PHI_ADJ_temp = solver.solve()
        PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_temp_reshaped = PostProcessor.postprocess_power2DHexx(PHI_ADJ_temp, conv_tri, group, N_hexx)

        output = {"keff": keff.real}
        for g in range(len(PHI_ADJ_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS_reshaped = globals().get("dSIGS_reshaped")
        dNUFIS = globals().get("dNUFIS")
        noise_section = globals().get("noise_section")
        os.makedirs(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}', exist_ok=True)

        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
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

        dPOWER = []
        dRHO = []

        freq = np.logspace(-4, 4, num=101)

        for f in range(len(freq)):
            ff = freq[f]
            print(f'Solving for frequency {ff:.3e}')
            omega = 2 * np.pi * ff

            dTOT_hexx = expand_XS_hexx(group, J_max, I_max, dTOT, level)
            dSIGS_hexx = expand_SIGS_hexx(group, J_max, I_max, dSIGS_reshaped, level)
            chi_hexx = expand_XS_hexx(group, J_max, I_max, chi, level)
            dNUFIS_hexx = expand_XS_hexx(group, J_max, I_max, dNUFIS, level)

            if noise_section == 1:
                # Collect all non-zero indices of dTOT_hexx for each group
                for g in range(group):
                    for n in range(N_hexx):
                        if dTOT_hexx[g][n] != 0:
                            noise_tri_index = n//(6 * (4 ** (levels - 1))) * (6 * (4 ** (levels - 1))) + 3
                            if n != noise_tri_index:
                                dTOT_hexx[g][n] = 0
            else:
                pass
            if type_noise == 'FVX' or type_noise == 'FAV':
                if levels != 4:
                    print('Vibrating Assembly type noise only works if level = 4. Changing level to 4')
                    levels = 4

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
    
            with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_freq{f}_output.json', 'w') as json_file:
                json.dump(output, json_file, indent=4)

            v1_PP = v1
            v2_PP = v2
            v_new = [v1_PP, v2_PP]

            NUMER = 0
            DENOM = 0
            for g in range(group):
                NUMER += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * dPHI_temp_reshaped[g]), dx=h**2/4*np.sqrt(3), axis = 0)
                DENOM += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * PHI_temp_reshaped[g]), dx=h**2/4*np.sqrt(3), axis = 0)

            dPOWER.append(NUMER / DENOM)

            S_NUMER = dS.dot(PHI)
            S_NUMER_reshaped = np.reshape(S_NUMER, (group, max(conv_tri)))
            S_DENOM = F_FORWARD.dot(PHI)
            S_DENOM_reshaped = np.reshape(S_DENOM, (group, max(conv_tri)))
            S_NUMER_reshaped = np.nan_to_num(S_NUMER_reshaped, nan=0)
            S_DENOM_reshaped = np.nan_to_num(S_DENOM_reshaped, nan=0)

            NUMER = 0
            DENOM = 0
            for g in range(group):
                NUMER += trapezoid(PHI_ADJ_temp_reshaped[g] * S_NUMER_reshaped[g], dx=h**2/4*np.sqrt(3), axis = 0)
                DENOM += trapezoid(PHI_ADJ_temp_reshaped[g] * S_DENOM_reshaped[g], dx=h**2/4*np.sqrt(3), axis = 0)

            dRHO.append(NUMER / DENOM)

        G_0 = []
        mag_G_0 = []
        phase_G_0 = []
        for f in range(len(freq)):
            transfer_function = dPOWER[f]/dRHO[f]
            G_0.append(transfer_function)
            mag_G_0.append(abs(transfer_function))
            phase_G_0.append(np.degrees(np.angle(transfer_function)))

        # OUTPUT
        print(f'Generating JSON output')
        output = {}
        G_0_groupname = f'G_0'
        G_0_list = [{"real": x.real, "imaginary": x.imag} for x in G_0]
        output[G_0_groupname] = G_0_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_G_0_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Plotting magnitude
        plt.figure(figsize=(8, 6))  # Create a new figure
        plt.plot(freq, mag_G_0, marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude of Neutron Noise')
        plt.title('Plot of Transfer Function (Magnitude)')
        plt.savefig(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_Magnitude.png')

        # Plotting phase
        plt.figure(figsize=(8, 6))  # Create another new figure
        plt.plot(freq, phase_G_0, marker='o')
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylim(-120, 0)
        plt.ylabel('Phase of Neutron Noise')
        plt.title('Plot of Transfer Function (Phase)')
        plt.savefig(f'{output_dir}/{case_name}_TRANSFER/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_Phase.png')

if __name__ == "__main__":
    main()