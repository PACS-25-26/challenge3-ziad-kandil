#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "Grid.hpp"

class Solver{

    public:

    /**
     * @brief Initializes the grid with the given values.
     */
    static void initialize(Grid& grid);

    /**
     * @brief Performs one iteration of the Jacobi method to update the grid values.
     */
    static void jacobi_step(Grid& grid);

    /**
     * @brief Computes the local error of the current grid values compared to the previous iteration.
     * @return The computed local error as a double value.
     */
    static double compute_local_error(const Grid& grid);

    /**
     * @brief Computes the exact solution of the problem at the given coordinates (x, y).
     * @param x The x-coordinate for which to compute the exact solution.
     * @param y The y-coordinate for which to compute the exact solution.
     * @return The computed exact solution as a double value.
     */
    static double exact_solution(double x, double y);

    /**
     * @brief Computes the forcing term of the problem at the given coordinates (x, y).
     * @param x The x-coordinate for which to compute the forcing term.
     * @param y The y-coordinate for which to compute the forcing term.
     * @return The computed forcing term as a double value.
     */
    static double forcing_term(double x, double y);

    /**
     * @brief Computes the local L2 error of the current grid values compared to the exact solution.
     * @return The computed local L2 error as a double value.
     */
    static double compute_local_l2_error(const Grid& grid, const std::vector<double>& solution);
};


#endif