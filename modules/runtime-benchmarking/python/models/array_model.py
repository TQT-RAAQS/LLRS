from collections import namedtuple
from os import stat

import matplotlib.pyplot as plt
import numpy as np

from random import shuffle

from structures.geometry import ArrayGeometry, Params
from models.trap_model import Trap
from models.atom_model import Atom
from structures.graph import DynamicArrayGraph
# from experiment.modules.LLRS.src.operational.utils.emulator_utils import LossParams


# Definition of a tuple containing a trap's euclidean coordinates and Trap object
Coordinate = namedtuple('Coordinate', ['x', 'y', 'z'])

class Array(object):
    '''
    Defines the geometry of a spatial distribution of objects.

    Attributes:
    ----
        type : str
            type of geometry, inherited from Geometry object

        coordinates : list of np.ndarray
            coordinates of traps within trap array, inherited from Geometry object

        coordinates_before_shift : list of np.ndarray
            coordinates of traps within trap array before shift to centre of mass, inherited from Geometry object

        sampling_step : list
            in which format traps are populated along each generator vector, inherited from Geometry object

        basis : np.ndarray
            basis vectors of trap array (if lattice/sub-lattice geometry), inherited from Geometry object

    '''


    def __init__(self, geometry: ArrayGeometry):
        '''
        Creates a new TrapArray object

        Args
        ----
            geometry (Geometry) : Geometry class or Lattice class, specify the geometry of the TrapArray object
            PSF (np.ndarray) : point-spread function
            PSF_factor (np.ndarray) : 

        '''

        self.geometry = geometry

        self.type = geometry.type
        self.coordinates = geometry.coordinates
        self.coordinates_before_shift = geometry.coordinates_before_shift
        self.shift = geometry.coordinates[0] - geometry.coordinates_before_shift[0]
        self.trap_number_along_generators = geometry.trap_number
        self.sampling_step = geometry.sampling_step
        self.array_object_by_idx = {}

        self._random_seed = 0
        if geometry.basis is not None:
            self.basis = geometry.basis

    def get_lattice(self):
        '''
        Returns a 3D np.array of locations of the trap array. Each row is the (x, y, z) of one trap.

        Returns:
        ----
            lattice (np.ndarray) : array of shape (n_atoms, 3) of lattice points in 3D space

        '''

        traps = self.array_object_by_idx.values()
        lattice = np.zeros((len(traps), 3))
        for idx, t in enumerate(traps):
            lattice[idx] = np.array([[t.euclidean_coords[0], t.euclidean_coords[1], t.euclidean_coords[2]]])

        return lattice

    def get_cm_shift(self):
        '''
        Returns the centre of mass shift vector of the entire trap array.

        Returns:
        ----
            (list) : list of x, y, z coordinates of centre of mass shift

        '''

        trans_before_shift = self.coordinates_before_shift.T
        total_trap = len(self.coordinates_before_shift)
        mid_x = sum(trans_before_shift[0]) / total_trap
        mid_y = sum(trans_before_shift[1]) / total_trap
        mid_z = sum(trans_before_shift[2]) / total_trap
        return [mid_x, mid_y, mid_z]

    def __len__(self):
        return len(self.array_object_by_idx.keys())

class TrapArray(Array):
    '''
    Defines the geometry of a spatial distribution of traps.

    Attributes:
    ----
        type : str
            type of geometry, inherited from Geometry object

        coordinates : list of np.ndarray
            coordinates of traps within trap array, inherited from Geometry object

        coordinates_before_shift : list of np.ndarray
            coordinates of traps within trap array before shift to centre of mass, inherited from Geometry object

        sampling_step : list
            in which format traps are populated along each generator vector, inherited from Geometry object

        array_object_by_idx : dict
            dict of traps via trap indices, inherited from Geometry object

        trap_number : list
            list of traps along each generator, inherited from Geometry object

        basis : np.ndarray
            basis vectors of trap array (if lattice/sub-lattice geometry), inherited from Geometry object

        PSF : 

        PSF_factor : 

    '''


    def __init__(self, geometry: ArrayGeometry, PSF: np.ndarray = None, PSF_factor: np.ndarray = None):
        '''
        Creates a new TrapArray object

        Args
        ----
            geometry (Geometry) : Geometry class or Lattice class, specify the geometry of the TrapArray object
            PSF (np.ndarray) : point-spread function
            PSF_factor (np.ndarray) : 

        '''
        
        self.idx_by_coords = {}
        self.trap_number = 0

        super().__init__(geometry)
        self.PSF = PSF
        self.PSF_factor = PSF_factor

        for i in range(len(geometry.coordinates)):
            coords = np.around(geometry.coordinates[i], 14)
            if len(coords) == 2:
                coords = (coords[0], coords[1], 0)

            trap = Trap(coords)
            location = Coordinate(*coords)
            idx = self.trap_number

            self.idx_by_coords[location] = idx
            self.array_object_by_idx[idx] = trap

            self.trap_number += 1

    def get_trap_by_index(self, index):
        '''
        Returns a trap given an index

        Args:
        ----
            index (int) : index of desired trap

        Returns:
        ----
            (Trap) : Trap in array indexed by a label

        '''

        return self.array_object_by_idx[index]

    def get_trap_by_coordinates(self, coordinates):
        '''
        Returns a trap given its coordinates

        Args:
        ----
            coordinates (np.ndarray) : (3,) array of trap coordinates

        Returns:
        ----
            (Trap) : Trap in array indexed by its coordinates

        '''

        location = Coordinate(*coordinates)
        return self.array_object_by_idx[self.idx_by_coords[location]]

    def get_trap_indices(self):
        '''
        Returns the indices of all traps

        Returns:
        ----
            (list) : list of trap indices

        '''

        return list(self.array_object_by_idx.keys())

    def get_occupied_traps_coords(self):
        '''
        Returns the coordinates of traps that are filled.

        Returns:
        ----
            (np.ndarray) : array of shape (n_atoms, 3) of coordinates of traps that contain an atom

        '''

        coords = []
        for trap in self.get_trap_indices():
            trap = self.get_trap_by_index(trap)
            if trap.atoms is not None:
                coords.append(trap.euclidean_coords)
        
        return np.asarray(coords)

    def get_occupation_state(self):
        '''
        Returns a list of labels of traps that are filled.

        Returns:
        ----
            (list) : list of traps that contain an atom

        '''

        indices = []
        for trap_idx in self.get_trap_indices():
            trap = self.get_trap_by_index(trap_idx)
            if len(list(trap.atoms)) > 0:
                indices.append(trap_idx)
        
        return indices

    def get_occupation_list(self) -> np.ndarray:
        '''
        Returns a list of 0 or 1 which represents the array 
        occupation state according to trap index.
        '''

        array = np.zeros(self.trap_number, dtype=int)

        for idx in self.get_occupation_state():
            array[idx] = 1

        return array

    def set_occupation_state(self, filling_type: str, n_atoms: int = 0, trap_coordinates : list = [], trap_indices : list = [], loss_params=None):
        '''
        Set the occupation state of the trap array.

        Args:
        ----
            atom_array (AtomArray) : array of atoms distributed in space within a static trap array

        Raises: 
        ----
            Exception: for invalid coordinates, trap indices

        '''

        for trap in self.get_trap_indices():
            self.get_trap_by_index(trap).clear_trap()

        if filling_type == 'full':
            for i, trap_idx in enumerate(self.get_trap_indices()):
                trap = self.get_trap_by_index(trap_idx)
                atom = Atom(trap.euclidean_coords, loss_params, id_=i)
                self.trap_atom(atom, coords=trap.euclidean_coords)

        elif filling_type == 'random':
                trap_keys = list(self.get_trap_indices())
                shuffle(trap_keys)
                for i, trap_idx in enumerate(trap_keys[:n_atoms]):
                    trap = self.get_trap_by_index(trap_idx)
                    atom = Atom(trap.euclidean_coords, loss_params, id_=i)
                    self.trap_atom(atom, coords=trap.euclidean_coords)

        elif filling_type == 'coordinates':
            for given_coord in trap_coordinates:
                try:
                    for coord in self.coordinates:
                        if np.all(np.isclose(coord, given_coord, 1e-5)):
                            self.get_trap_by_coordinates(tuple(coord)).load_trap()
                            break
                except:
                    raise Exception('Coordinates provided that are not in the trap array.')
        elif filling_type == 'indices':
            traps = self.get_trap_indices()
            for i, trap_idx in enumerate(trap_indices):
                if trap_idx in traps:
                    trap = self.get_trap_by_index(trap_idx)
                    atom = Atom(trap.euclidean_coords, loss_params, id_=i)
                    self.trap_atom(atom, coords=trap.euclidean_coords)
                else:
                    raise Exception('Trap index provided that are not in geometry.')
        else:
            raise Exception('Please provide valid filling type. Valid options are: {full, random, coordinates, indices}.')

    def clear_all_traps(self) -> None:
        '''
        
        '''

        for idx in self.get_trap_indices():
            self.get_trap_by_index(idx).clear_trap()

    def trap_atom(self, atom, trap_idx=None, coords=None):
        if trap_idx is not None:
            trap = self.get_trap_by_index(trap_idx)
        elif coords is not None:
            trap = self.get_trap_by_coordinates(coords)
        else:
            Exception('No trap identifier provided.')
        trap.trap_atom(atom)

    def visualize(self, fig, ax, labels: bool=False, 
                  trap_size: int=200, fill_size: int=75,
                  trap_marker: str='o', trap_edgewidth=0.5, atom_marker: str='o',
                  atomclr: str='black', edgeclr: str='black', edges=None, edgecolor='orange', view_all_traps=False, 
                  recolour_target=False, atomclr_target='#57B52C', target=[]):
        '''
        Plot the state of the trap array

        '''

        lattice = self.get_lattice()

        dimensions = 3
        dim_sum = 0
        for i in range(3):
            dim_sum += (np.min(lattice[:, i]) == np.max(lattice[:, i]))
        if dim_sum <= 2:
            dimensions = 2

        # TODO: should it be using self.coordinates instead?
        plot_lattice = np.transpose(lattice)

        x_active = []
        y_active = []
        x_filled = []
        y_filled = []
        #labels = self.get_atom_indices().values

        if dimensions == 2:
            if edges != None:
                for edge in edges:
                    pos1 = self.get_trap_by_index(edge[0]).euclidean_coords[:2]
                    pos2 = self.get_trap_by_index(edge[1]).euclidean_coords[:2]
                    ax.plot([pos1[0], pos2[0]], [pos1[1], [pos2[1]]], c=edgecolor)

            for i in range(len(plot_lattice[0])):
                x = plot_lattice[0][i]
                y = plot_lattice[1][i]

                if self.array_object_by_idx[i].get_status() or view_all_traps:
                    x_active.append(x)
                    y_active.append(y)
                if len(self.array_object_by_idx[i].atoms) > 0:
                    atom = list(self.array_object_by_idx[i].atoms)[0]
                    alpha = atom.get_corruption(0)
                    if recolour_target:
                        coords = atom.euclidean_coords

                        if any((target[:]==coords).all(1)):
                            ax.scatter(x, y, marker=atom_marker, linewidth=trap_edgewidth, facecolor=atomclr_target, edgecolor='none', 
                                    s=fill_size, alpha=alpha)
                        else:
                            ax.scatter(x, y, marker=atom_marker, linewidth=trap_edgewidth, facecolor=atomclr, edgecolor='none', 
                                    s=fill_size, alpha=alpha)
                    else:
                        ax.scatter(x, y, marker=atom_marker, linewidth=trap_edgewidth, facecolor=atomclr, edgecolor='none', 
                                s=fill_size, alpha=alpha)
            
            for pos in ['right', 'top', 'bottom', 'left']:
                fig.gca().spines[pos].set_visible(False)

            ax.scatter(x_active, y_active, marker=trap_marker, linewidth=trap_edgewidth, facecolor='none', 
                    edgecolor=edgeclr, s=trap_size)
                
            if labels:
                ax.text(x_active + 2e-7, y_active + 1e-7, str(i), fontsize=9)
            #ax.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        elif dimensions == 3:
            for i in range(len(plot_lattice[0])):
                x = plot_lattice[0][i]
                y = plot_lattice[1][i]
                z = plot_lattice[2][i]
                if self.array_object_by_idx[i].status:
                    ax.scatter(x, y, color='red')
                else:
                    ax.scatter(x, y, color='green')
                if labels:
                    ax.text(x + 0.03, y + 0.03, z + 0.03, i, fontsize=9)
            ax.axis('equal')

        #fig.tight_layout()



class StaticTrapArray(TrapArray):
    '''
    Interface for SLM traps.
    '''

    def __init__(self, geometry: ArrayGeometry, PSF: np.ndarray = None, PSF_factor: np.ndarray = None):
        super().__init__(geometry=geometry)
        for idx in self.array_object_by_idx.keys():
            self.array_object_by_idx[idx].set_on()

    def extract_trap(self, trap_label, dynamic_array):
        atoms = self.array_object_by_idx[trap_label].clear_trap()
        coords = self.array_object_by_idx[trap_label].euclidean_coords
        dynamic_trap = dynamic_array.get_trap_by_coordinates(coords)
        if len(atoms) > 0:
            dynamic_trap.undergo_extraction_operation()
        if dynamic_trap.get_status():
            for atom in atoms:
                atom.undergo_alpha_operation()
                dynamic_array.trap_atom(atom, coords=coords)
        else:
            Exception('Extracted atom to dynamic array trap that is OFF.')

    def print_trap_contents(self):
        atoms = []
        for idx, trap in self.array_object_by_idx.items():
            atoms.append(trap.get_num_atoms())
        for i in range(0, len(atoms), 4):
                    print(atoms[i:i+4])


class DynamicTrapArraySpace(TrapArray):
    '''
    Interface for AOD trap space.
    '''

    def __init__(self, geometry: ArrayGeometry, PSF: np.ndarray = None, PSF_factor: np.ndarray = None):
        super().__init__(geometry=geometry)

        self.graph = DynamicArrayGraph(array_object_by_idx=self.array_object_by_idx)
        # set graph edges, does nothing if arbitrary array
        self.set_dynamic_trap_array_graph_edges()

        for idx in self.array_object_by_idx.keys():
            self.array_object_by_idx[idx].set_off()

    def implant_trap(self, trap_label, static_array):
        trap = self.array_object_by_idx[trap_label]
        atoms = trap.clear_trap()
        if len(atoms) > 0:
            trap.undergo_implantation_operation()
        coords = trap.euclidean_coords
        
        # if trap is active
        if self.get_trap_by_coordinates(coords).get_status():
            for atom in atoms:
                atom.undergo_alpha_operation()
                static_array.trap_atom(atom, trap_idx=trap_label)
                

        else:
            Exception('Extracted atom to dynamic array trap that is OFF.')

    def set_dynamic_trap_array_graph_edges(self, edges: dict={}) -> None:
        '''
        Set or generate the edges between each vertex (trap) in the graph generated by the static trap array.

        Args:
        ----
            edges (list of 2-element sets) : list of edges of graph

        Raises:
        ----
            Exception: for passing an edge that contains a vertex not in the static trap array.

        '''

        if self.type == 'arbitrary':
            indices = self.array_object_by_idx.keys()
            for vertex in edges.keys():
                if vertex not in indices:
                    Exception('Invalid edge provided.')

        else:
            edges = dict()
            length_1 = self.trap_number_along_generators[0]
            length_2 = self.trap_number_along_generators[1]
            for index in self.array_object_by_idx.keys():
                edge_right = ((index+1)%length_1 == 0)
                edge_bottom = (index//(length_1) == length_2-1)
                if not edge_bottom:
                    if index in edges.keys():
                        edges[index].append(index + length_1)
                    else:
                        edges[index] = [index + length_1]
                    if (index + length_1) in edges.keys():
                        edges[index + length_1].append(index)
                    else:
                        edges[index + length_1] = [index]

                if not edge_right:
                    if index in edges.keys():
                        edges[index].append(index + 1)
                    else:
                        edges[index] = [index + 1]
                    if (index + 1) in edges.keys():
                        edges[index + 1].append(index)
                    else:
                        edges[index + 1] = [index]
        
        self.graph.set_edges(edges)

    def get_graph(self) -> DynamicArrayGraph:
        '''
        Return edges and vertices of a the trap array graph.

        Returns:
        ----
            V (np.ndarray of shape (n_traps, 3)) : list of trap coordinates index aligned with their labelling
            E (list of 2-element tuples) : list of edges within the graph

        '''

        return self.graph

