#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "Grid.hpp"

#include <functional>

class Solver{

    public:

    /**
     * @brief Initializes the grid with the given values.
     * @param grid The local grid structure 
     * @param f The forcing function (right-hand side)
     */
    static void initialize(Grid& grid, std::function<double(double, double)> f);

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
     * @brief Computes the local L2 error of the current grid values compared to the exact solution.
     * @param grid The local grid structure
     * @param solution The current solution vector to compare against the exact solution
     * @param exact_sol A function that computes the exact solution at given coordinates (x, y)
     * @return The computed local L2 error as a double value.
     */
    static double compute_local_l2_error(const Grid& grid, const std::vector<double>& solution, std::function<double(double, double)> exact_sol);
};


#endif