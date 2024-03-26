/*
 * Atom Class to be used for Operation Benchmarking in Main
 */

#ifndef _ATOM_H_
#define _ATOM_H_

#include <cmath>

/// defines State enum, representing whether the atom was implanted or extracted
enum State {
    IMPLANTED = 0,
    EXTRACTED = 1
};

/// defines time per move in seconds
#define SECONDS_PER_MOVE 10e-6

class Atom {
    // Atom class used to store properties and observables during reconfiguration sequence.

    // Attributes:
    //     n_alpha : int
    //         number of transfer operations undergone by atom
    //     n_nu : int
    //         number of displacement operations undergone by atom
    int n_alpha;
    int n_nu;
    State state;
    double alpha_loss;
    double nu_loss;
    double lifetime;

public:
    /**
     * @brief Constructor of the Atom Class
     * @param n_alpha number of transfer operations undergone by the atom
     * @param n_nu number of displacement operations undergone by the atom
     * @param state initial state of the atom (default: IMPLANTED)
     * @param alpha_loss probability for atom to be lost when transferred (default: 0.985)
     * @param nu_loss probability for atom to be lost when displaced (default: 0.985)
     * @param lifetime lifetime of the atom in seconds (default: 60.0)
     */
    Atom(int n_alpha , int n_nu , State state,
         double alpha_loss , double nu_loss , double lifetime );

    /**
     * @brief Function to handle transfer of atom from the static to dynamic array. Updates atom's number of transfers and switches its state
     * @param num_transfer number of transfers (default: 1)
     */
    void transfer(int num_transfer = 1);

    /**
     * @brief Updates specific atom's number of displacements within the dynamic array
     * @param num_displace number of displacements (default: 1)
     */
    void displace(int num_displace = 1);

    /**
     * @brief Getter function for number of transfers
     * @return Number of transfers
     */
    int getAlpha();

    /**
     * @brief Getter function for number of displacements
     * @return Number of displacements
     */
    int getNu();

    /**
     * @brief Getter function for atom lifetime
     * @return Lifetime
     */
    double getLifetime();

    /**
     * @brief Getter function for alpha loss
     * @return Probability for atom to be lost when transferred
     */
    double getAlphaLoss();

    /**
     * @brief Getter function for nu loss
     * @return Probability for atom to be lost when displaced
     */
    double getNuLoss();

    /**
     * @brief Getter function for the state of the atom
     * @return Variable describing whether the atom was implanted or extracted
     */
    State getState();

    /**
     * @brief Sets the state of the atom
     * @param new_state New state to set for the atom
     */
    void setState(State new_state);

    /**
     * @brief Calculates and returns the probability that the atom will be lost
     * @param total_moves Total number of moves performed
     * @return Probability of atom loss
     */
    double getLoss(int total_moves = 0);

    /**
     * @brief Resets number of transfers and displacements
     */
    void clearCorruption();
};

#endif
