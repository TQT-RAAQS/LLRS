/*
* Trap Array Class to be used for Operation Benchmarking in Main
*/

#ifndef _TRAP_ARRAY_H_
#define _TRAP_ARRAY_H_

#include <vector>
#include <tuple>
#include <bits/stdc++.h>
#include <random>
#include "atom.h"
#include "Solver.h"
#include "llrs-lib/Settings.h"

class TrapArray {
    // Trap Array Class used to store configurations of atoms in traps
        int Nt_x; // number of rows
        int Nt_y; // numbers of columns
        double loss_params_alpha = 0.985;
        double loss_params_nu = 0.985;
		double lifetime = 60.0;
        int total_moves = 0;
        std::vector<std::vector<Atom *>> traps; // vector of traps
    public:
        /**
         * @brief Constructor of the TrapArray Class
         * @param Nt_x              Number of rows of the TrapArray
         * @param Nt_y              Number of columns of the TrapArray
         * @param current_config    Binary vector indicating the presence of atoms
         * @param loss_params_alpha Loss parameter alpha (default = 0)
         * @param loss_params_nu    Loss parameter nu (default = 0)
         * @param total_moves       Total number of moves (default = 0)
         */
        TrapArray(int Nt_x, int Nt_y, const std::vector<int32_t> &current_config, double loss_params_alpha = 0, double loss_params_nu = 0, double lifetime = 60, int total_moves = 0);

        /**
         * @brief Destructor for TrapArray class
         */
        ~TrapArray();

        int reverse(int new_Nt_y);
        
        /**
         * @brief Function to check if a given move is in the Bounds of Trap Array class
         * @param low_Nt_x Low limit of rows
         * @param high_Nt_x High limit of rows
         * @param low_Nt_y Low limit of columns
         * @param high_Nt_y High limit of columns
         * @return Error code (1 if out of bounds)
         */
        
        int checkBounds(int low_Nt_x, int high_Nt_x, int low_Nt_y, int high_Nt_y);

        /**
         * @brief Function to check if a Vertical Movement is "legal"
         * @param Nt_x1 The row corresponding to the current position of the trap
         * @param Nt_y1 The column corresponding to the current position of the trap
         * @param Nt_y2 The column corresponding to the future position of the trap
         * @return Error code (1 if collision)
         */
        int checkMovementVertical(int Nt_x1, int Nt_y1, int Nt_y2);

        /**
         * @brief Function to check if a Horizontal Movement is "legal"
         * @param Nt_x1 The row corresponding to the current position of the trap
         * @param Nt_x2 The row corresponding to the future position of the trap
         * @param Nt_y1 The column corresponding to the current position of the trap
         * @return Error code (1 if collision)
         */
        int checkMovementHorizontal(int Nt_x1, int Nt_x2, int Nt_y1);

        /**
         * @brief Function to move atoms based on a move type
         * @param low_Nt_x     The row corresponding to the current move
         * @param low_Nt_y     The column corresponding to the current move
         * @param block_size   The size corresponding to the current move
         * @param move_type    The type of move
         * @return Error code (2 if collision)
         */
        int moveAtoms(int low_Nt_x, int low_Nt_y, int block_size, Synthesis::WfMoveType move_type);


        /**
         * @brief Perform a list of moves on a Trap Array
         * @param moves_list The list of moves to perform
         * @return Error code (1 if collision)
         */
        int performMoves(std::vector<Reconfig::Move> &moves_list);

        /**
         * @brief Return the total number of moves performed by the trap_array
         * @return Total number of moves
         */
        int getTotalMoves();

        /**
         * @brief Increase the total number of moves performed by the trap array
         * @param num_moves The number of moves to increase by (default 1)
         */
        void increaseTotalMoves(int num_moves = 1);

        /**
         * @brief Perform loss on the Trap Array based on the moves on each atom and lifetime
         */
        void performLoss();

        /**
         * @brief Edit a binary vector to represent the current configuration of Atoms in the Trap Array
         * @param current_config Vector to be edited
         */
        void getArray(std::vector<int32_t> &current_config);

        /**
         * @brief Print the current configuration in the Trap Array
         * @param Nt_x1 Low limit of rows to print (default = 0)
         * @param Nt_x2 High limit of rows to print (default = 0)
         * @param Nt_y1 Low limit of columns to print (default = 0)
         * @param Nt_y2 High limit of columns to print (default = 0)
         */
        void printTrapArray(int Nt_x1 = 0, int Nt_x2 = 0, int Nt_y1 = 0, int Nt_y2 = 0);

        /**
         * @brief Print the current bounds in the Trap Array
         * @param Nt_x1 Low limit of rows to print (default = 0)
         * @param Nt_x2 High limit of rows to print (default = 0)
         * @param Nt_y1 Low limit of columns to print (default = 0)
         * @param Nt_y2 High limit of columns to print (default = 0)
         */
        void printTrapArrayBounds(int Nt_x1 = 0, int Nt_x2 = 0, int Nt_y1 = 0, int Nt_y2 = 0);


        void printTraps();
        int getRelaventMoves();
};

#endif
