/**
 * @brief TrapArray class deals with the optical trap array and movements of atoms between the traps
 * @date Dec 2023
*/

#include "trap-array.h"

#define ATOM_EXISTS_CONFIG 1
#define NO_ATOM nullptr 


int TrapArray::reverse(int new_Nt_y) {
    return Nt_y - new_Nt_y - 1;
}

/**
 * @brief Constructor of the TrapArray Class
 * @param width number of rows of the TrapArray
 * @param height number of cols of the TrapArray
 * @param current_config binary vector indicating the presence of atoms
*/

TrapArray::TrapArray(int Nt_x, int Nt_y, const std::vector<int32_t> &current_config, double loss_params_alpha, double loss_params_nu, double lifetime,int total_moves): Nt_x(Nt_x), Nt_y(Nt_y), loss_params_alpha(loss_params_alpha), loss_params_nu(loss_params_nu), lifetime(lifetime), total_moves(total_moves) {
    
    traps.resize(Nt_x, std::vector<Atom *>(Nt_y, nullptr));
    //Add Atom objects to traps where an atom exists

    for (int32_t i  = 0; i < current_config.size(); i++) {
        if (current_config[i] == ATOM_EXISTS_CONFIG) {
            traps[i % Nt_x][i / Nt_x] = new Atom(0,0, IMPLANTED, loss_params_alpha, loss_params_nu, lifetime);
        } else {

            traps[i / Nt_y][i % Nt_y] = NO_ATOM;
        }
    }


}
/**
 * @brief Destructor for TrapArray class
*/
TrapArray::~TrapArray() {
    for (auto &it: traps) {
        for (auto &it2: it) {
            delete it2;
        }
    }
}

/**
 * @brief Function to check if a given move is in the Bounds of Trap Array class
 * The parameters are self explanatory
*/
int TrapArray::checkBounds(int low_Nt_x, int high_Nt_x, int low_Nt_y, int high_Nt_y) {
    if (high_Nt_x >= this->Nt_x || low_Nt_x < 0) {
        std::cout << low_Nt_x << " " << high_Nt_x << std::endl;
        std::cout << "OUT OF RANGE ROW ERROR" << std::endl;
        return 1;
    }
    if (high_Nt_y >= this->Nt_y || low_Nt_y < 0) {
        std::cout << low_Nt_y << " " << high_Nt_y << std::endl;
        std::cout << "OUT OF RANGE COL ERROR" << std::endl;
        return 1;
    }
    return 0;
}

/**
 * @brief Function to check if a Vertical Movement is "legal"
 * @param row1 The row corresponding to the current position of the trap
 * @param row2 The row corresponding to the future position of the trap
 * @param col The column corresponding to the current and future position of the trap
 * @return error code
*/

int TrapArray::checkMovementVertical(int Nt_x1, int Nt_y1, int Nt_y2) {
    std::cout << "Nt_x1 " << Nt_x1 << " Nt_y1 " << Nt_y1 << " Nt_y2 " << Nt_y2 << std::endl;
    std::cout << traps.size() << std::endl;
    if ((traps.at(Nt_x1)).at(Nt_y1) != NO_ATOM && (traps.at(Nt_x1)).at(Nt_y2) != NO_ATOM) {
        this->printTrapArray(Nt_x1-1, Nt_x1+1, Nt_y1-1, Nt_y2+1);
        std::cout << "VERTICAL: " << Nt_x1 << " " << Nt_y1 << " " << Nt_y2 << std::endl;
        std::cout << "ATOM COLLISION" << std::endl;
        return 1;
    }
    std::cout << "Made it through" << std::endl;
    return 0;
}

/**
 * @brief Function to check if a Horizontal Movement is "legal"
 * @param row The row corresponding to the current and future position of the trap
 * @param col1 The column corresponding to the current position of the trap
 * @param col2 The column corresponding to the future position of the trap
 * @return error code
*/

int TrapArray::checkMovementHorizontal(int Nt_x1, int Nt_x2, int Nt_y1) {

    //Check if there is an existing atom in the target trap, if there is then there is a collision

    if (traps.at(Nt_x1).at(Nt_y1) != NO_ATOM && traps.at(Nt_x2).at(Nt_y1) != NO_ATOM) {
        this->printTrapArray(Nt_x1-1, Nt_x2+1, Nt_y1-1, Nt_y1+1);
        std::cout << "HORIZONTAL: " << Nt_x1 << " " << Nt_x2 << " " << Nt_y1 << std::endl;
        std::cout << "ATOM COLLISION" << std::endl;
        return 1;
    }

    return 0;
}

/**
 * @brief Function to move atoms based on a move type
 * @param low_row The row corresponding to the current move
 * @param low_col The column corresponding to the current move
 * @param block_size The size corrsponding to the current move (number of atoms to move)
 * @param move_type The type of move
*/
int TrapArray::moveAtoms(int low_Nt_x, int low_Nt_y, int block_size, Synthesis::WfMoveType move_type) {
    switch (move_type) { 
        case Synthesis::FORWARD_1D:
        case Synthesis::UP_2D:
        {
            std::cout << "in switch" << std::endl;
            int high_Nt_y = low_Nt_y + block_size;
            for (int i = high_Nt_y-1; i >= low_Nt_y; --i) {
                //check for atom collision
                std::cout<<" "<<low_Nt_x<<" "<<i<<std::endl;
                if (checkMovementVertical(low_Nt_x, i, i+1) != 0) {
                    return 2;
                }
                //if target is free and the source trap has an atom, move the atom from the source to target
                if (this->traps[low_Nt_x][i] != NO_ATOM) {
                    this->traps[low_Nt_x][i]->displace();
                    auto tmp = this->traps[low_Nt_x][i];
                    this->traps[low_Nt_x][i] = this->traps[low_Nt_x][i+1];
                    this->traps[low_Nt_x][i+1] = tmp;
                }
            }
        }
        break;
        case Synthesis::BACKWARD_1D: 
        case Synthesis::DOWN_2D:
        {
            int high_Nt_y = low_Nt_y - block_size;
            for (int i = low_Nt_y+1; i <= high_Nt_y; ++i) {
                //check for atom collision
                if (checkMovementVertical(low_Nt_x, i, i-1) != 0) {
                    return 2;
                }
                //if target is free and the source trap has an atom, move the atom from the source to target            
 
                if (this->traps[low_Nt_x][i] != NO_ATOM) {
                    this->traps[low_Nt_x][i]->displace();
                    auto tmp = this->traps[low_Nt_x][i];
                    this->traps[low_Nt_x][i] = this->traps[low_Nt_x][i-1];
                    this->traps[low_Nt_x][i-1] = tmp;
                }
            }
        }
        break;
        case Synthesis::RIGHT_2D:
        {
            int new_Nt_x = low_Nt_x + 1;
            //check for atom collision
            if (checkMovementHorizontal(low_Nt_x, new_Nt_x, low_Nt_y) != 0) {
                return 2;
            }
            //if target is free and the source trap has an atom, move the atom from the source to target            
            if (this->traps[low_Nt_x][low_Nt_y] != NO_ATOM) {
                this->traps[low_Nt_x][low_Nt_y]->displace();
                auto tmp = this->traps[low_Nt_x][low_Nt_y];
                this->traps[low_Nt_x][low_Nt_y] = this->traps[new_Nt_x][low_Nt_y];
                this->traps[new_Nt_x][low_Nt_y] = tmp;
            }
        }
        break;
        case Synthesis::LEFT_2D:
        {
            int new_Nt_x = low_Nt_x;
            int old_Nt_x = low_Nt_x + 1;
            //check for atom collision
            if (checkMovementHorizontal(old_Nt_x, new_Nt_x, low_Nt_y) != 0) {
                return 2;
            }
            //if target is free and the source trap has an atom, move the atom from the source to target
            if (this->traps[old_Nt_x][low_Nt_y] != NO_ATOM) {
                this->traps[old_Nt_x][low_Nt_y]->displace();
                auto tmp = this->traps[old_Nt_x][low_Nt_y];
                this->traps[old_Nt_x][low_Nt_y] = this->traps[new_Nt_x][low_Nt_y];
                this->traps[new_Nt_x][low_Nt_y] = tmp;
            }
        }
        break;
    }
    return 0;
}


/**
 * @brief Perform a list of 1D moves on a Trap Array
 * @param moves_list The list of moves to perform
*/
int TrapArray::performMoves(std::vector<Reconfig::Move> &moves_list) {
    for (const auto &move: moves_list) {
        increaseTotalMoves(1);
        Synthesis::WfMoveType move_type = std::get<0>(move);
        int row = std::get<1>(move);
        int col = std::get<2>(move);     // 0 if 1D
        int length = std::get<3>(move);  // block_size

        switch (move_type){
            case Synthesis::EXTRACT_1D:
                for (int i = row; i < length + row; ++i) {
                    if (this->traps[i][col] == NO_ATOM) {
                        continue;
                    }
                    
                    this->traps[i][col]->transfer();
                    this->traps[i][col]->setState(EXTRACTED);
                }
                break;
            
            case Synthesis::IMPLANT_1D:
                for (int i = row; i < length + row; ++i) {
                    if (this->traps[i][col] == NO_ATOM) {
                        continue;
                    }
                    this->traps[i][col]->transfer();
                    this->traps[i][col]->setState(IMPLANTED);
                }
                break;
            case Synthesis::EXTRACT_2D:
                // if (this->checkBounds(new_Nt_x, new_Nt_x, new_Nt_y, new_Nt_y + length - 1)) {
                //   return 1;
                // }
                for (int i = row; i < length + row; ++i) {
                    if (this->traps[col][i] == NO_ATOM) {
                        continue;
                    }
                    this->traps[col][i]->transfer();
                    this->traps[col][i]->setState(EXTRACTED);
                }
                break;
            
            case Synthesis::IMPLANT_2D:
                //  if (this->checkBounds(new_Nt_x, new_Nt_x, new_Nt_y, new_Nt_y + length - 1)) {
                //     return 1;
                // }
                for (int i = row; i < length + row; ++i) {
                    if (this->traps[col][i] == NO_ATOM) {
                        continue;
                    }

                    this->traps[col][i]->transfer();
                    this->traps[col][i]->setState(IMPLANTED);
                }
                break;
    
            default:
                std::cout<<"row "<<row << " col "<<col<< " length " << length << " move type " << move_type <<std::endl;
                int ret = this->moveAtoms(row, col, length, move_type);
                if (ret != 0) {
                    return 1;
                }
        }
    }
    return 0;
}


/**
 * @brief Return the total number of moves performed by the trap_array
*/
int TrapArray::getTotalMoves() {
    return this->total_moves;
}

/**
 * @brief Increase the total number of moves performed by the trap array
 * @param num_moves The number of moves to increase by (default 1)
*/

void TrapArray::increaseTotalMoves(int num_moves) {
    this->total_moves += num_moves;
    return;
}

/**
 * @brief Perform loss on the Trap Array based on the moves on each atom and lifetime
*/

void TrapArray::performLoss() {
    double counter = 0;
    int total = 0;
    // std::srand(0);
    for (size_t i = 0; i < this->Nt_x; ++i) {
        for (size_t j = 0; j < this->Nt_y; ++j) {
            if (this->traps[i][j] == NO_ATOM) {
                continue;
            }
            double loss_value = this->traps[i][j]->getLoss(this->getTotalMoves());
            double random = (double)(((double)rand()) / (double) RAND_MAX);
            total += 1;
            counter += loss_value;
            if (loss_value < random) {
                delete this->traps[i][j];
                this->traps[i][j] = NO_ATOM;
            } else {
                this->traps[i][j]->clearCorruption();
            }
        }
    }
    this->total_moves = 0;
    return;
}

/**
 * @brief Edit a binary vector to represent current configuration of Atoms in the Trap Array
*/

void TrapArray::getArray(std::vector<int32_t> &current_config) {
    if (current_config.size() != (this->Nt_x * this->Nt_y)) {
        std::cout << "THE DIMENSIONS DO NOT MATCH" << std::endl;
        return;
    }
    for (size_t i = 0; i < this->Nt_x; ++i) {
        for (size_t j = 0; j < this->Nt_y; ++j) {
            if (this->traps[i][j] != NO_ATOM) {
                current_config[j*this->Nt_x + i] = 1;
            }
            else {
                current_config[j*this->Nt_x + i] = 0;
            }
        }
    }
    return;
}

/**
 * @brief Print the current configuration in the Trap Array
*/

void TrapArray::printTrapArray(int Nt_x1, int Nt_x2, int Nt_y1, int Nt_y2) {
    if (Nt_x1+Nt_x2+Nt_y1+Nt_y2 == 0) {
        this->printTrapArrayBounds(0, this->Nt_x, 0, this->Nt_y);
    }
    else {
        this->printTrapArrayBounds(max(Nt_x1, 0), min(Nt_x2+1, this->Nt_x) , max(Nt_y1, 0), min(Nt_y2+1, this->Nt_y));
    }
    return;
}

/**
 * @brief Print the current bounds the Trap Array
*/

void TrapArray::printTrapArrayBounds(int Nt_x1, int Nt_x2, int Nt_y1, int Nt_y2) {
    std::cout << "   ";
    for (int i = Nt_x1; i < Nt_x2; ++i) {
        std::cout << i << "          ";
    }
    std::cout << std::endl;
    for (int i = Nt_x1; i < Nt_x2; ++i) {
        std::cout << "------------";
    }
    std::cout << std::endl;
    for (int j = Nt_y1; j < Nt_y2; ++j) {
        if (j >= 10) {
            std::cout << j <<  "|";
        }
        else {
            std::cout << j <<  "| ";
        }
        for (int i = Nt_x1; i < Nt_x2; ++i) {
            if (this->traps[i][j] == NO_ATOM) {
                std::cout << "0: (0, 0), ";
            }
            else if (this->traps[i][j]->getState() == IMPLANTED) {
                std::cout << "I: (" << this->traps[i][j]->getAlpha();
                std::cout << ", " << this->traps[i][j]->getNu() << "), ";
            }
            else {
                std::cout << "E: (" << this->traps[i][j]->getAlpha();
                std::cout << ", " << this->traps[i][j]->getNu() << "), ";
            }
        }
        std::cout << std::endl;
    }
}



