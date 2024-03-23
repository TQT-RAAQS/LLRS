from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sympy


class Lattice(object):
    def __init__(self, lattice_type, trap_number: list = []):
        '''
        Creates a new Lattice object for TrapArray object. Subclass of Geometry

        Parameters
        ----------
        lattice_type:
            string, is one of 'linear lattice', 'rectangular lattice', 'square lattice', 'oblique',
            'centered rectangular lattice', 'triangular lattice', 'hexagonal lattice', 'cyclic group', 'graphene'

        trap_number:
            number of traps on each direction of basis (for lattices) e.g. [int1, int2, int3] representing the
            number of traps along k1, k2, k3 respectively
        '''

        self.type = lattice_type
        self.trap_number = trap_number

    def line_lattice_coefficient_matrix(self):
        return self.primitive_rectangular_lattice_coefficient_matrix()

    def primitive_rectangular_lattice_coefficient_matrix(self):
        if len(self.trap_number) == 3:
            total_trap = self.trap_number[0] * self.trap_number[1] * self.trap_number[2]
            coefficient_matrix = np.zeros((total_trap, 4))
            # k for k3(z), j for k2(y), i for k1(x)
            for k in range(self.trap_number[2]):
                for j in range(self.trap_number[1]):
                    for i in range(self.trap_number[0]):
                        idx = k * self.trap_number[0] * self.trap_number[1] + \
                              j * self.trap_number[0] + i
                        coefficient_matrix[idx] = np.array([i, j, k, 1])
            # print(coefficient_matrix)
            return coefficient_matrix
        else:
            raise ValueError('Invalid dimension, higher than 3D.')

    def centered_rectangular_lattice_coefficient_matrix(self):
        if len(self.trap_number) == 3:
            nx = self.trap_number[0]
            ny = self.trap_number[1]
            nz = self.trap_number[2]
            xy_total = nx * ny + (nx - 1) * (ny - 1)
            total_trap = xy_total * nz
            coefficient_matrix = np.zeros((total_trap, 4))
            # j for k2(y), i for k1(x)
            for k in range(nz):
                for j in range(ny * 2 - 1):
                    if j % 2 == 0:
                        for i in range(nx):
                            idx = k * xy_total + (2 * nx - 1) * (j // 2) + i
                            coefficient_matrix[idx] = np.array([i, j / 2, k, 1])
                    else:
                        for i in range(nx - 1):
                            idx = k * xy_total + (2 * nx - 1) * (j // 2) + nx + i
                            coefficient_matrix[idx] = np.array([i + 1 / 2, j / 2, k, 1])
            return coefficient_matrix

    def triangular_lattice_coefficient_matrix(self):
        if len(self.trap_number) == 3:
            if self.trap_number[0] == self.trap_number[1]:
                xy_total = sum(range(self.trap_number[0] + 1))
                total_trap = xy_total * self.trap_number[2]
                coefficient_matrix = np.zeros((total_trap, 4))
                row_trap_number = self.trap_number[0]
                for k in range(self.trap_number[2]):
                    for j in range(self.trap_number[1]):
                        for i in range(row_trap_number):
                            idx = k * xy_total + sum(range(self.trap_number[0], row_trap_number, -1)) + i
                            coefficient_matrix[idx] = np.array([i, j, k, 1])
                        row_trap_number = row_trap_number - 1
                    row_trap_number = self.trap_number[0]
                return coefficient_matrix
            else:
                raise ValueError('Triangular Lattice cannot be generated. Need to have same number of points')

    def hexagonal_lattice_coefficient_matrix(self):
        if len(self.trap_number) == 3:
            total_trap = self.trap_number[0] * self.trap_number[1] * self.trap_number[2]
            coefficient_matrix = np.zeros((total_trap, 4))
            # j for k2(y), i for k1(x)
            for k in range(self.trap_number[2]):
                for j in range(self.trap_number[1]):
                    for i in range(self.trap_number[0]):
                        idx = k * self.trap_number[0] * self.trap_number[1] + j * self.trap_number[0] + i
                        coefficient_matrix[idx] = np.array([j // 2 + i, j, k, 1])
            return coefficient_matrix

    def cyclic_group_coefficient_matrix(self):
        # The number of traps is the number of atoms that should be placed on a circle
        # Two basis vectors should be of same length
        if len(self.trap_number) == 3:
            total_trap = self.trap_number[0] * self.trap_number[2]
            coefficient_matrix = np.zeros((total_trap, 4))
            for k in range(self.trap_number[2]):
                for i in range(self.trap_number[0]):
                    theta = 0 + 2 * np.pi * i / self.trap_number[0]
                    x_coordinate = np.cos(theta)
                    y_coordinate = np.sin(theta)
                    idx = k * self.trap_number[0] + i
                    coefficient_matrix[idx] = np.array([x_coordinate, y_coordinate, k, 1])
            return coefficient_matrix

    def graphene_coefficient_matrix(self):
        nx = self.trap_number[0]
        ny = (self.trap_number[1] + 1) * 2
        nz = self.trap_number[2]
        if ny % 4 == 0:
            xy_total = ((2 * nx + 1) * (ny // 2))
            total_trap = xy_total * nz
        elif ny % 4 == 2:
            xy_total = ((2 * nx + 1) * (ny // 2 - 1) + 2 * nx - 1)
            total_trap = xy_total * nz
        coefficient_matrix = np.zeros((total_trap, 4))
        for k in range(nz):
            for j in range(ny):
                if j % 4 == 1 and j + 1 == ny:
                    for i in range(nx - 1):
                        idx = xy_total * k + nx + (4 * nx + 2) * (j - 1) // 4 + i
                        y_coordinates = self.trap_number[1] * 3 + 1
                        coefficient_matrix[idx] = np.array([1 + 2 * i, y_coordinates, k, 1])
                elif j % 4 == 0:
                    for i in range(nx):
                        idx = xy_total * k + (4 * nx + 2) * (j // 4) + i
                        y_coordinates = (j // 4) * 3 * 2
                        coefficient_matrix[idx] = np.array([2 * i, y_coordinates, k, 1])
                elif j % 4 == 1:
                    for i in range(nx + 1):
                        idx = xy_total * k + (4 * nx + 2) * (j // 4) + nx + i
                        y_coordinates = (j // 4) * 3 * 2 + 1
                        coefficient_matrix[idx] = np.array([-1 + 2 * i, y_coordinates, k, 1])
                elif j % 4 == 2:
                    for i in range(nx + 1):
                        idx = xy_total * k + (4 * nx + 2) * (j // 4) + 2 * nx + 1 + i
                        y_coordinates = (j // 4) * 3 * 2 + 3
                        coefficient_matrix[idx] = np.array([-1 + 2 * i, y_coordinates, k, 1])
                else:
                    for i in range(nx):
                        idx = xy_total * k + (4 * nx + 2) * (j // 4) + 3 * nx + 2 + i
                        y_coordinates = (j // 4) * 3 * 2 + 4
                        coefficient_matrix[idx] = np.array([2 * i, y_coordinates, k, 1])
        return coefficient_matrix


class SubLattice(object):
    def __init__(self, lattice_type, sampling_step: tuple = ()):
        '''
        Creates a new Lattice object for TrapArray object. Subclass of Geometry

        Parameters
        ----------
        lattice_type:
            string, is one of 'rectangular sublattice', 'triangular sublattice', 'kagome'
            'quasi-crystal'

        sampling_step:
            a tuple of array (non uniformly distributed) steps to generate sublattices
        '''

        # self.dimension = dimension
        self.type = lattice_type
        self.sampling_step = sampling_step

    def line_sublattice_coefficient_matrix(self):
        return self.rectangular_sublattice_coefficient_matrix()

    def rectangular_sublattice_coefficient_matrix(self):
        if len(self.sampling_step) == 3:
            x_sample = np.zeros(len(self.sampling_step[0]) + 1)
            x_sample[1:] = self.sampling_step[0]
            LX = len(x_sample)
            y_sample = np.zeros(len(self.sampling_step[1]) + 1)
            y_sample[1:] = self.sampling_step[1]
            LY = len(y_sample)
            z_sample = np.zeros(len(self.sampling_step[2]) + 1)
            z_sample[1:] = self.sampling_step[2]
            LZ = len(z_sample)
            total_trap = LX * LY * LZ
            coefficient_matrix = np.zeros((total_trap, 4))
            for k in range(LZ):
                z_coordinate = sum(z_sample[0:k + 1])
                for j in range(LY):
                    y_coordinate = sum(y_sample[0:j + 1])
                    for i in range(LX):
                        x_coordinate = sum(x_sample[0:i + 1])
                        idx = k * LX * LY + j * LX + i
                        coefficient_matrix[idx] = np.array([x_coordinate, y_coordinate, z_coordinate, 1])
            return coefficient_matrix

        else:
            raise ValueError('Invalid dimension, not 2D or 3D.')

    def triangular_sublattice_coefficient_matrix(self):
        if len(self.sampling_step) == 3:
            if len(self.sampling_step[0]) == len(self.sampling_step[1]):
                matrix = np.zeros((2, len(self.sampling_step[0])))
                matrix[0] = self.sampling_step[0]
                matrix[1] = self.sampling_step[1]
                _, diag = sympy.Matrix(matrix).rref()

                if len(diag) == 1:
                    x_sample = np.zeros(len(self.sampling_step[0]) + 1)
                    x_sample[1:] = self.sampling_step[0]
                    LX = len(x_sample)
                    y_sample = np.zeros(len(self.sampling_step[1]) + 1)
                    y_sample[1:] = self.sampling_step[1]
                    LY = len(y_sample)
                    z_sample = np.zeros(len(self.sampling_step[2]) + 1)
                    z_sample[1:] = self.sampling_step[2]
                    LZ = len(z_sample)
                    total_trap = sum(range(LX + 1)) * LZ
                    coefficient_matrix = np.zeros((total_trap, 4))
                    row_trap_number = LX

                    for k in range(LZ):
                        z_coordinate = sum(z_sample[0:k + 1])
                        for j in range(LY):
                            y_coordinate = sum(y_sample[0:j + 1])
                            for i in range(row_trap_number):
                                # x_coordinate = sum(x_sample[0:i+1])
                                if i == 0:
                                    x_coordinate = 0
                                else:
                                    x_coordinate = sum(x_sample[j + 1:j + i + 1])
                                idx = k * sum(range(LX + 1)) + sum(range(LX, row_trap_number, -1)) + i
                                coefficient_matrix[idx] = np.array([x_coordinate, y_coordinate, z_coordinate, 1])
                            row_trap_number = row_trap_number - 1
                        row_trap_number = LX
                    return coefficient_matrix

    def kagome_coefficient_matrix(self):
        main_row = len(self.sampling_step[0]) + 1
        min_row = int(main_row / 2)
        total_row = 2 * main_row + 1
        row_max_idx = 2 * main_row
        row_trap_number = np.zeros(total_row, dtype=int)
        step = self.sampling_step[0][0] / 2
        y_coordinate = np.zeros(total_row)
        x_start = np.zeros(total_row)
        for i in range(main_row + 1):
            y_coordinate[i] = (main_row - i) * step
            y_coordinate[row_max_idx - i] = (i - main_row) * step
            if i % 2 == 0:
                row_trap_number[i] = min_row + i // 2
                row_trap_number[row_max_idx - i] = min_row + i // 2
                x_start[i] = (1 - i) * step
                x_start[row_max_idx - i] = (1 - min_row * 2) * step
            else:
                row_trap_number[i] = min_row * 2 + i + 1
                row_trap_number[row_max_idx - i] = min_row * 2 + i + 1
                x_start[i] = (-i) * step
                x_start[row_max_idx - i] = -main_row * step

        z_sample = np.zeros(len(self.sampling_step[2]) + 1)
        z_sample[1:] = self.sampling_step[2]
        LZ = len(z_sample)
        total_trap = sum(row_trap_number) * LZ
        coefficient_matrix = np.zeros((total_trap, 4))
        for k in range(LZ):
            z_coordinate = sum(z_sample[0:k+1])
            for j in range(len(row_trap_number)):
                for i in range(row_trap_number[j]):
                    if j % 2 == 0:
                        x_coordinate = x_start[j] + 2 * step * i
                    else:
                        x_coordinate = x_start[j] + step * i
                    idx = k * sum(row_trap_number) + (sum(row_trap_number[0:j]) + i)
                    coefficient_matrix[idx] = np.array([x_coordinate, y_coordinate[j], z_coordinate, 1])
        return coefficient_matrix

