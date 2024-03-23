from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin, pi, sqrt
from .lattice import Lattice
import sympy
import params as pp
import difflib
from copy import deepcopy

class Params(pp.Params):
    trap_number = pp.Param([], doc="The number of traps along x, y, z axis respectively", dtype=list)
    norms = pp.Param(2 * 1e-6, doc="norm of k1, k2, k3, number or list of length 3 [norm_k1, norm_k2, norm_k3]")
    sampling_index = pp.Param([], doc="The sampling points along x, y, z axis respectively, list of 3 lists, "
                                       "start from 1 or tuple of length 3. If type is kagome, then give "
                                       "(edge length of hexagon, number, z sampling)", dtype=list)
    z = pp.Param(0, doc="Longitudinal shift if sampling step is given in [x, y]")
    shift_vector = pp.Param([0, 0, 0], doc="The origin of the geometry", dtype=list)
    k1 = pp.Param([1, 0, 0], doc="Basis vector k1", dtype=list)
    k2 = pp.Param([0, 1, 0], doc="Basis vector k2", dtype=list)
    k3 = pp.Param([0, 0, 1], doc="Basis vector k3", dtype=list)
    euler_angle = pp.Param(([0, 0, 0],"rad"), doc="Euler Angle: [alpha, beta, gamma], unit: deg or rad", dtype=tuple)
    coordinates = pp.Param([], doc="coordinates of traps, list of np.array", dtype=list)


class GeometryOperation(object):
    """
    Creates a new Geometry object for TrapArray object (mainly for arbitrary geometry)

    Parameters
    ----------
    coordinates:
        array of coordinates(1D, 2D or 3D) for arbitrary traps e.g np.array([a1,b1,c1],[a2,b2,c2],...)

    geometry_type:
        string, "arbitrary" by default, unless other specified
    """

    def __init__(self, coordinates, geometry_type="arbitrary"):
        self.type = geometry_type
        self.coordinates = []

        for coordinate in coordinates:
            self.add_coordinate(coordinate)

    def add_coordinate(self, coordinate):
        if len(self.coordinates) == 0:
            self.coordinates = np.array([coordinate])
        elif len(self.coordinates) > 0 and len(self.coordinates[0]) != len(coordinate):
            raise ValueError("The coordinate provided is not consistent with the dimension of the geometry")
        else:
            self.coordinates = np.append(self.coordinates, np.array([coordinate]), axis=0)

    # remove a coordinate by its value
    def remove_coordinate(self, coordinate: np.array):
        """
        Removes a single trap by coordinate
        """
        self.coordinates.remove(coordinate)

    def remove_coordinates_at(self, indices: List[int]):
        """
        Removes a series of coordinates by their array_object_by_idx
        Parameters
        ----------
        indices: a list of indices to remove
        """
        self.coordinates = np.delete(self.coordinates, indices)

    def visualize(self):
        plot_lattice = np.transpose(self.coordinates)
        if len(self.coordinates[0]) == 3:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for i in range(len(plot_lattice[0])):
                x = plot_lattice[0][i]
                y = plot_lattice[1][i]
                z = plot_lattice[2][i]
                ax.scatter(x * 1e6, y * 1e6, z * 1e6, color="blue")
                ax.text(x * 1e6 + 0.05, y * 1e6 + 0.05, z * 1e6 + 0.05, i,
                        fontsize=9, marker='o', facecolors='none', edgecolors='r')
            plt.title("Trap Geometry: " + self.type.capitalize())
            ax.set_xlabel("X (\u03bcm)")
            ax.set_ylabel("Y (\u03bcm)")
            ax.set_zlabel("Z (\u03bcm)")
            ax.set_aspect('equal')
            plt.show()

class ArrayGeometry(object):
    def __init__(self, lattice_type: str, params: Params):

        self.type = standardize_str(lattice_type)
        self.shift_vector = self.generate_shift_vector(params)
        self.trap_number = self.generate_trap_number(params)
        self.sampling_step = self.generate_sampling_step(params)
        self.basis = self.generate_basis(params)
        self.rotation_matrix = self.generate_rotation_matrix(params)
        self.generator_matrix = self.get_generator_matrix(params)
        self.coordinates_before_shift = self.generate_coordinates_before_shift()
        self.coordinates = self.generate_lattice()

    @staticmethod
    def generate_shift_vector(params):
        shift = np.zeros(3)
        shift[:len(params.shift_vector)] = np.asarray(params.shift_vector)
        return shift + np.array([0, 0, params.z])

    def generate_trap_number(self, params):
        if self.type.lower() == "cyclic group":
            if len(params.trap_number) == 1:
                return [params.trap_number[0], 1, 1]
            elif len(params.trap_number) == 2:
                return [params.trap_number[0], 1, params.trap_number[1]]
            else:
                raise Exception("cyclic group only needs number of traps in xy plane and z (number of layers)")
        if len(params.trap_number) == 1:
            return [params.trap_number[0], 1, 1]
        elif len(params.trap_number) == 2:
            return [params.trap_number[0], params.trap_number[1], 1]
        else:
            return params.trap_number

    def generate_sampling_step(self, params):
        if params.sampling_index == []:
            return []
        elif len(params.sampling_index) == 1:
            sampling_points_x = params.sampling_index[0]
            sampling_points_y = np.array([1])
            sampling_points_z = np.array([1])
            if not all(isinstance(x, int) for x in sampling_points_x):
                raise Exception("Check sampling points at x or k1 direction, integers required.")
            x_sampling_step = np.diff(sampling_points_x)
            y_sampling_step = np.diff(sampling_points_y)
            z_sampling_step = np.diff(sampling_points_z)
            return (x_sampling_step, y_sampling_step, z_sampling_step)
        elif len(params.sampling_index) == 2:
            if self.type.lower() == "kagome":
                if params.sampling_index[1] % 2 == 0:
                    raise Exception("Invaid number of hexagon on main diagonal")
                x_sampling_step = np.ones(params.sampling_index[1]) * 2 * params.sampling_index[0]
                y_sampling_step = np.ones(params.sampling_index[1]) * 2 * params.sampling_index[0]
                sampling_points_z = np.array([1])
                if not all(isinstance(x, int) for x in sampling_points_z):
                    raise Exception("Check sampling points at z or k3 direction, integers required.")
                z_sampling_step = np.diff(sampling_points_z)
                return (x_sampling_step, y_sampling_step, z_sampling_step)
            else:
                sampling_points_x = params.sampling_index[0]
                sampling_points_y = params.sampling_index[1]
                sampling_points_z = np.array([1])
                if not all(isinstance(x, int) for x in sampling_points_x):
                    raise Exception("Check sampling points at x or k1 direction, integers required.")
                if not all(isinstance(x, int) for x in sampling_points_y):
                    raise Exception("Check sampling points at y or k2 direction, integers required.")
                x_sampling_step = np.diff(sampling_points_x)
                y_sampling_step = np.diff(sampling_points_y)
                z_sampling_step = np.diff(sampling_points_z)
                return (x_sampling_step, y_sampling_step, z_sampling_step)
        elif len(params.sampling_index) == 3:
            if self.type.lower() == "kagome":
                if params.sampling_index[1] % 2 == 0:
                    raise Exception("Invaid number of hexagon on main diagonal")
                x_sampling_step = np.ones(params.sampling_index[1]) * 2 * params.sampling_index[0]
                y_sampling_step = np.ones(params.sampling_index[1]) * 2 * params.sampling_index[0]
                sampling_points_z = params.sampling_index[2]
                if not all(isinstance(x, int) for x in sampling_points_z):
                    raise Exception("Check sampling points at z or k3 direction, integers required.")
                z_sampling_step = np.diff(sampling_points_z)
                return (x_sampling_step, y_sampling_step, z_sampling_step)
            else:
                sampling_points_x = params.sampling_index[0]
                sampling_points_y = params.sampling_index[1]
                sampling_points_z = params.sampling_index[2]
                if not all(isinstance(x, int) for x in sampling_points_x):
                    raise Exception("Check sampling points at x or k1 direction, integers required.")
                if not all(isinstance(x, int) for x in sampling_points_y):
                    raise Exception("Check sampling points at y or k2 direction, integers required.")
                if not all(isinstance(x, int) for x in sampling_points_z):
                    raise Exception("Check sampling points at z or k3 direction, integers required.")
                x_sampling_step = np.diff(sampling_points_x)
                y_sampling_step = np.diff(sampling_points_y)
                z_sampling_step = np.diff(sampling_points_z)
                return (x_sampling_step, y_sampling_step, z_sampling_step)

    def generate_basis(self, params):
        # obtain normalized basis
        vec1 = np.array(params.k1) / np.linalg.norm(params.k1)
        vec2 = np.array(params.k2) / np.linalg.norm(params.k2)
        vec3 = np.array(params.k3) / np.linalg.norm(params.k3)
        if self.type.lower() in ["kagome", "hexagonal lattice"]:
            vec1 = np.array([1, 0, 0])
            vec2 = np.array([-1/2, sqrt(3)/2, 0])
        if self.type.lower() == "graphene":
            vec1 = np.array([sqrt(3)/2, 0, 0])
            vec2 = np.array([0, 1/2, 0])
        norms = params.norms
        if len(params.trap_number) == 1 or len(params.sampling_index) == 1:
            basis = (vec1 * norms, np.array([0, 0, 0]), np.array([0, 0, 0]))
            return basis
        elif len(params.trap_number) == 2 or len(params.sampling_index) == 2:
            if isinstance(norms, float) or isinstance(norms, int):
                basis = (vec1 * norms, vec2 * norms, np.array([0, 0, 1]))
                if self.type.lower() == "cyclic group":
                    basis = (vec1 * norms, vec2 * norms, vec3 * norms)
                return basis
            elif isinstance(norms, list) and len(norms) == 2:
                basis = (vec1 * norms[0], vec2 * norms[1], np.array([0, 0, 0]))
                if self.type.lower() == "cyclic group":
                    basis = (vec1 * norms[0], vec2 * norms[1], vec3 * norms[2])
                return basis
            else:
                raise Exception("Please check the dimension of basis length.")
        elif len(params.trap_number) == 3 or len(params.sampling_index) == 3:
            if isinstance(norms, float) or isinstance(norms, int):
                basis = (vec1 * norms, vec2 * norms, vec3 * norms)
                return basis
            elif isinstance(norms, list) and len(norms) == 3:
                basis = (vec1 * norms[0], vec2 * norms[1], vec3 * norms[2])
                return basis
            else:
                raise Exception("Please check the dimension of basis length.")
        else:
            raise Exception("Please check the dimension, the acceptable dimension is 1d, 2d, or 3d")

    @staticmethod
    def generate_rotation_matrix(params):
        alpha = params.euler_angle[0][0]
        beta = params.euler_angle[0][1]
        gamma = params.euler_angle[0][2]
        unit = params.euler_angle[1]
        if unit.lower() == "deg":
            alpha = np.deg2rad(alpha)
            beta = np.deg2rad(beta)
            gamma = np.deg2rad(gamma)
        R_z = np.array([[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]])
        R_y = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
        R_x = np.array([[1, 0, 0], [0, cos(gamma), -sin(gamma)], [0, sin(gamma), cos(gamma)]])
        R = R_z @ R_y @ R_x
        return R.T

    def get_generator_matrix(self, params):
        shift_vector = np.array(params.shift_vector)
        basis = self.basis
        generator_matrix = np.array([basis[0], basis[1], basis[2], shift_vector])
        return generator_matrix

    def generate_coordinates_before_shift(self):
        if self.type == "line lattice":
            line = Lattice(self.type, self.trap_number)
            coefficient_matrix = line.line_lattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type in ["rectangular lattice", "square lattice", "oblique"]:
            rectangular = Lattice(self.type, self.trap_number)
            coefficient_matrix = rectangular.primitive_rectangular_lattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "centered rectangular lattice":
            centered_rectangular = Lattice(self.type, self.trap_number)
            coefficient_matrix = centered_rectangular.centered_rectangular_lattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "triangular lattice":
            triangular = Lattice(self.type, self.trap_number)
            coefficient_matrix = triangular.triangular_lattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "hexagonal lattice":
            hexagonal = Lattice(self.type, self.trap_number)
            coefficient_matrix = hexagonal.hexagonal_lattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "cyclic group":
            cyclic = Lattice(self.type, self.trap_number)
            coefficient_matrix = cyclic.cyclic_group_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "graphene":
            graphene = Lattice(self.type, self.trap_number)
            coefficient_matrix = graphene.graphene_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "quasi-crystal":
            return
        elif self.type == "line sublattice":
            line_sublattice = SubLattice(self.type, self.sampling_step)
            coefficient_matrix = line_sublattice.line_sublattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "rectangular sublattice":
            rectangular_sublattice = SubLattice(self.type, self.sampling_step)
            coefficient_matrix = rectangular_sublattice.rectangular_sublattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "triangular sublattice":
            triangular_sublattice = SubLattice(self.type, self.sampling_step)
            coefficient_matrix = triangular_sublattice.triangular_sublattice_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        elif self.type == "kagome":
            kagome = SubLattice(self.type, self.sampling_step)
            coefficient_matrix = kagome.kagome_coefficient_matrix()
            lattice = coefficient_matrix @ self.generator_matrix @ self.rotation_matrix
            return lattice
        else:
            raise ValueError("The input lattice/sublattice type is undefined.")

    def generate_lattice(self):
        coordinates = deepcopy(self.coordinates_before_shift)
        total_trap = len(coordinates)
        trans_before_shift = coordinates.T
        mid_x = sum(trans_before_shift[0]) / total_trap
        mid_y = sum(trans_before_shift[1]) / total_trap
        mid_z = sum(trans_before_shift[2]) / total_trap
        origin = self.shift_vector - np.array([mid_x, mid_y, mid_z])
        for i in range(total_trap):
            coordinates[i] = coordinates[i] + origin
        return coordinates

    def vertical_view(self):
        plot_lattice = np.transpose(self.coordinates)
        for i in range(len(plot_lattice[0])):
            x = plot_lattice[0][i]
            y = plot_lattice[1][i]
            plt.scatter(x * 1e6, y * 1e6, marker='o', facecolors='none', edgecolors='b')
            #plt.text(x * 1e6 + 0.05, y * 1e6 + 0.05, i, fontsize=9)
        plt.xlabel("X (\u03bcm)")
        plt.ylabel("Y (\u03bcm)")
        #plt.gca().set_aspect(1.0/get_data_ratio())
        #plt.gca().set_aspect('equal')
        plt.axis("equal")
        plt.show()

    def visualize(self):
        plot_lattice = np.transpose(self.coordinates)
        if len(self.coordinates[0]) == 3:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for i in range(len(plot_lattice[0])):
                x = plot_lattice[0][i]
                y = plot_lattice[1][i]
                z = plot_lattice[2][i]
                ax.scatter(x * 1e6, y * 1e6, z * 1e6, marker='o', facecolors='none', edgecolors='b')
                ax.text(x * 1e6 + 0.05, y * 1e6 + 0.05, z * 1e6 + 0.01, i, fontsize=9)
            plt.title("Trap Geometry: " + self.type.capitalize())
            ax.set_xlabel("X (\u03bcm)")
            ax.set_ylabel("Y (\u03bcm)")
            ax.set_zlabel("Z (\u03bcm)")
            #ax.set_aspect('equal')
            plt.show()


def standardize_str(string):
    geometry_keywords = ["line", "rectangular", "rect", "square", "squ", "oblique", "obl", "centered", "cent",
                         "triangular", "trig", "hexagonal", "hex", "cyclic", "cyc", "group", "graphene", "graph",
                         "kagome", "kag", "lattice", "sublattice", "sub"]
    geo_type = string.lower().split()
    input_list = []
    for s in geo_type:
        item = difflib.get_close_matches(s, geometry_keywords)
        if item != []:
            input_list.append(item[0])
    input_list = list(dict.fromkeys(input_list))
    lattice_cond = "lattice" in input_list
    sublattice_cond = ("sublattice" in input_list) or ("sub" in input_list)
    L = len(input_list)
    if (not lattice_cond) and (not sublattice_cond):
        if L == 1:
            if "oblique" in input_list or "obl" in input_list:
                return "oblique"
            elif "graphene" in input_list or "graph" in input_list:
                return "graphene"
            elif "kagome" in input_list or "kag" in input_list:
                return "kagome"
            elif "cyclic" in input_list or "cyc" in input_list:
                return "cyclic group"
        elif L == 2:
            if ("cyclic" in input_list or "cyc" in input_list) and ("group" in input_list):
                return "cyclic group"
        else:
            return "Cannot detect input Geometry type"
    elif lattice_cond and (not sublattice_cond):
        input_list.remove("lattice")
        L = len(input_list)
        if L == 1:
            if "line" in input_list:
                return "line lattice"
            elif "centered" in input_list or "cent" in input_list:
                return "centered rectangular lattice"
            elif "rectangular" in input_list or "rect" in input_list:
                return "rectangular lattice"
            elif "square" in input_list or "squ" in input_list:
                return "square lattice"
            elif "triangular" in input_list or "trig" in input_list:
                return "triangular lattice"
            elif "hexagonal" in input_list or "hex" in input_list:
                return "hexagonal lattice"
            elif "cyclic" in input_list or "cyc" in input_list:
                return "cyclic group"
        elif L == 2:
            if ("centered" in input_list or "cent" in input_list) and \
                    ("rectangular" in input_list or "rect" in input_list):
                return "centered rectangular lattice"
        else:
            return "Cannot detect input Geometry type"
    elif sublattice_cond:
        if "lattice" in input_list:
            input_list.remove("lattice")
        if "sublattice" in input_list:
            input_list.remove("sublattice")
        if "sub" in input_list:
            input_list.remove("sub")
        L = len(input_list)
        if L == 1:
            if "line" in input_list:
                return "line sublattice"
            elif "rectangular" in input_list or "rect" in input_list:
                return "rectangular sublattice"
            elif "triangular" in input_list or "trig" in input_list:
                return "triangular sublattice"
        else:
            return "Cannot detect input Geometry type"
    else:
        return "Please clarify if a lattice or a sublattice is wanted"
