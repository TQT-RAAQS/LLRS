#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July 6

@author: Drimik Roy
"""
import numpy as np

class Atom:
    """
    Atom class used to store properties and observables during reconfiguration sequence.

    Attributes:
    ----
        id : int
            integer id of atom object
        
        initial_euclidean_coords : np.ndarray of shape (3,)
            initial coords of atom when loaded

        euclidean_coords : np.ndarray of shape (3,)
            coords of atom
        
        N_alpha : int
            number of transfer operations undergone by atom

        N_nu : int
            number of displacement operations undergone by atom
            
        loss_params : LossParams
            parameters used to calculate corruption of atom

        N_nu_per_move : List[int]
            number of displacement operations performed on this atom per transfer

    """

    def __init__(self, coords, loss_params, id_=None):
        if id_ is not None:
            self.id_ = id_
        self.initial_euclidean_coords = coords
        self.euclidean_coords = coords
        self.N_alpha = 0
        self.N_nu = 0
        self.loss_params = loss_params
        self.N_nu_per_move = []

    def set_id(self, id_):
        self.id_ = id_
    
    def undergo_alpha_operation(self):
        self.N_alpha += 1
        if self.N_nu != sum(self.N_nu_per_move):
            self.N_nu_per_move.append(self.N_nu - sum(self.N_nu_per_move))
                
    def undergo_nu_operation(self):
        self.N_nu += 1
    
    def reset_corruption(self):
        self.N_alpha = 0
        self.N_nu = 0

    def get_corruption(self, sequence_time):
        return (self.loss_params["p_alpha"]**self.N_alpha)*(self.loss_params["p_nu"]**self.N_nu)*np.exp(-sequence_time/self.loss_params["t_lifetime"])

    def project(self):
        if np.random.random > self.get_corruption():
            self.corruption = 0
            return True
        self.corruption = 1
        return False

    def get_id(self):
        return self.id_

