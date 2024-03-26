/**
 * @brief Atom class represents atoms used in the experiment
 * @date Dec 2023
*/

#include "atom.h"

/**
 * @brief Constructor of the Atom Class
 * @param n_alpha number of transfer operations undergone by the atom
 * @param n_nu number of displacement operations undergone by the atom
*/

Atom::Atom(int n_alpha, int n_nu, State state, double alpha_loss, double nu_loss, double lifetime) {
    this->n_alpha = n_alpha;
    this->n_nu = n_nu;
    this->state = state;
    this->alpha_loss = alpha_loss;
    this->nu_loss = nu_loss;
    this->lifetime = lifetime;
}

/**
 * @brief Function to handle transfer of atom from static to dynamic array. Updates atom's number of transfers and switches it's state
 * @param num_transfer number of transfers
*/

void Atom::transfer(int num_transfer) {
    this->n_alpha += num_transfer;
    this->state = State(1 - this->state);
    return;
}

/**
 * @brief Updates specific atom's number of displacements within the dynamic array
 * @param n_displace number of displacements
*/

void Atom::displace(int num_displace) {
    this->n_nu += num_displace;
    return;
}

/**
 * @brief Getter function for number of transfers 
 * @return Number of transfers
*/

int Atom::getAlpha() {
    return this->n_alpha;
}

/**
 * @brief Getter function for number of displacements 
 * @return Number of displacements
*/

int Atom::getNu() {
    return this->n_nu;
}

/**
 * @brief Getter function for atom lifetime
 * @return Lifetime
*/

double Atom::getLifetime() {
    return this->lifetime;
}

/**
 * @brief Getter function for alpha loss
 * @return probability for atom to be lost when transferred
*/

double Atom::getAlphaLoss() {
    return this->alpha_loss;
}

/**
 * @brief Getter function for nu loss
 * @return probability for atom to be lost when displaced
*/

double Atom::getNuLoss() {
    return this->nu_loss;
}

/**
 * @brief Getter function for the state of the atom
 * @return variable describing whether the atom was implanted or extracted
*/

State Atom::getState() {
    return this->state;
}

/**
 * @brief Constructor of the Atom Class
 * @param n_alpha number of transfer operations undergone by the atom
*/

void Atom::setState(State new_state) {
    this->state = new_state;
}

/**
 * @brief Calculates and returns the probability that the atom will be lost
 * @return probability of atom loss
*/

double Atom::getLoss(int total_moves) {
    return  std::pow(this->getAlphaLoss(), this->getAlpha()) *
            std::pow(this->getNuLoss(), this->getNu()) * 
            std::exp(((-1) * total_moves * SECONDS_PER_MOVE ) / this->getLifetime());
}

/**
 * @brief Resets number of transfers and displacements
*/

void Atom::clearCorruption() {
    this->n_alpha = 0;
    this->n_nu = 0;
}
