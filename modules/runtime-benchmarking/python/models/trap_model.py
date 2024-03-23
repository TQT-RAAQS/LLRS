#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Drimik Roy
'''
from __future__ import absolute_import

import numpy as np
from copy import deepcopy

class Trap:
    '''
    A potential well


    Attributes
    ----------
    atoms : set 
                Atom objects trapped
    euclidean_coords : tuple of reals, optional
                Trap location in euclidean_space
                default : None
    Note
    ----
    Methods such as _instance_trap_atom() are referenced as trap_atom() for a Trap object.
    There exists a static vectorized method called trap_atom that is for convenience
    but individual objects call trap_atom. This behavior is replicated with other methods in the class. 
    '''

    def __init__(self, euclidean_coords=None, PSF=None, aberration=None):
        self.atoms = set()
        if euclidean_coords is not None:
            self.euclidean_coords = euclidean_coords
        self._initialize_ftn_calls()
        self.status = False
        self.PSF = PSF
        self.aberration = aberration
        self.active = True
        self.n_nu = 0
        self.n_extract = 0
        self.n_implant = 0

    def _initialize_ftn_calls(self):
        '''
        Initialize function call indirection

        '''
        self.get_num_atoms = self._instance_get_num_atoms
        self.trap_atom = self._instance_trap_atom
        self.remove_atom = self._instance_remove_atom

    def get_status(self):
        return self.active

    def set_on(self):
        self.active = True

    def set_off(self):
        self.active = False

    def undergo_extraction_operation(self):
        self.n_extract += 1

    def undergo_implantation_operation(self):
        self.n_implant += 1

    def undergo_nu_operation(self):
        self.n_nu += 1

    @staticmethod
    @np.vectorize
    def trap_atom(trap, atom):
        '''
        Vectorized call for trapping atoms (convenience ftn)

        Parameters
        ----------
        atom: Atom object

        '''
        trap.trap_atom(atom)

    def _instance_trap_atom(self, atom):
        '''
        Add atom to trap

        Parameters
        ----------
        atom : Atom object

        Note
        ----
        any instance calls remove_atom

        '''
        if isinstance(atom, list) or isinstance(atom, set):                
            self.atoms.update(atom)
        else:
            self.atoms.add(atom)

    @staticmethod
    @np.vectorize
    def remove_atom(trap, atom):
        '''
        Vectorized call for removing atom (convenience ftn)

        Parameters
        ----------
        trap : Trap object
        atom : Atom object

        '''
        trap.remove_atom(atom)

    def _instance_remove_atom(self, atom):
        '''
        Remove atom from trap

        Parameters
        ----------
        atom : Atom object

        Note
        ----
        any instance calls remove_atom

        '''
        try:
            self.atoms.remove(atom)
        except KeyError:
            raise KeyError('removing atom not held in trap')

    def clear_trap(self):
        '''
        Remove and return atoms from this trap

        Returns
        -------
        atoms : list of atom objects from this trap

        '''
        atoms = set(self.atoms)
        self.atoms.clear()
        return atoms

    def clear_trap_with_id(self):
        '''
        Remove atoms from this trap and return ids of atoms in this trap

        Returns
        -------
        atom_ids : list of atom ids from this trap


        Raises
        ------
        AttributeError
            if any atom in the trap does not have id set

        '''
        atom_ids = [atom.get_id() for atom in self.atoms]
        self.atoms.clear()
        return atom_ids

    # getters

    @staticmethod
    @np.vectorize
    def get_num_atoms(trap):
        '''
        Vectorized call for get_num_atoms (convenience ftn)

        '''
        return trap.get_num_atoms()

    def _instance_get_num_atoms(self):
        '''
        Returns
        -------
        Get number of atoms in this trap

        Note
        ----
        any instance calls get_num_atoms

        '''
        return len(self.atoms)

    def get_atoms_trapped(self):
        '''
        Returns
        -------
        set of atom objects trapped

        '''
        return self.atoms

    # overloaded operators

    def __eq__(self, other):
        '''
        Returns
        -------
        Equality based on number of atoms in both traps

        Parameters
        ----------
        other : Trap object

        '''
        return len(self.atoms) == len(other.atoms)

    def __ne__(self, other):
        '''
        Returns
        -------
        Inequality based on number of atoms in both traps

        Parameters
        ----------
        other : Trap object

        '''
        return len(self.atoms) != len(other.atoms)
