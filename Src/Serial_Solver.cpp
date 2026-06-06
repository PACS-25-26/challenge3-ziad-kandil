#include "Serial_Solver.hpp"

#include <cmath>
#include <iostream>


/**
 * @brief Initializes the grid with the given values.
 * @param grid The grid structure 
 * @param f The forcing function (right-hand side)
 */
void Serial_Solver::initialize(Grid& grid, std::function<double(double, double)> f){

    for(int i = 0; i < grid.global_n; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = j * grid.h;
            double y_coord = i * grid.h;

            grid.rhs[grid.idx(i,j)] = f(x_coord, y_coord);
        }
    }
}

/**
 * @brief Performs one iteration of the Jacobi method to update the grid values.
 */
void Serial_Solver::jacobi_step(Grid& grid){

    const double h2 = grid.h * grid.h;

    for(int i = 1; i < grid.global_n - 1; ++i){

        for (int j = 1; j < grid.ny - 1; ++j){

            grid.u_new[grid.idx(i,j)] = 0.25 * (
                grid.u_old[grid.idx(i-1, j)] +
                grid.u_old[grid.idx(i+1, j)] +
                grid.u_old[grid.idx(i, j-1)] +
                grid.u_old[grid.idx(i, j+1)] +
                h2 * grid.rhs[grid.idx(i,j)]
            );
        }
    }
}


/**
 * @brief Computes the error of the current grid values compared to the previous iteration.
 * @return The computed error as a double value.
 */
double Serial_Solver::compute_error(const Grid& grid){

    double sum = 0.0;
    
    for(int i = 0; i < grid.global_n; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double diff = grid.u_new[grid.idx(i,j)] - grid.u_old[grid.idx(i,j)];
            sum += diff * diff;
        }
    }

    return std::sqrt(sum * grid.h);
}

/**
 * @brief Computes the L2 error of the current grid values compared to the exact solution.
 * @param grid The grid structure
 * @param solution The current solution vector to compare against the exact solution
 * @param exact_sol A function that computes the exact solution at given coordinates (x, y)
 * @return The computed L2 error as a double value.
 */
double Serial_Solver::compute_l2_error(const Grid& grid, const std::vector<double>& solution, std::function<double(double, double)> exact_sol){

    double sum = 0.0;

    for(int i = 0; i < grid.global_n; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = j * grid.h;
            double y_coord = i * grid.h;

            double exact_value = exact_sol(x_coord, y_coord);
            double diff = solution[grid.idx(i,j)] - exact_value;
            sum += diff * diff;
        }
    }

    return std::sqrt(sum * grid.h);
}

/**
 * @brief Function to solve the equation in serial using the Jacobi method.
 * @param grid The grid structure containing the solution vectors and grid parameters.
 * @param tol The convergence tolerance for the iterative solver.
 * @param max_iters The maximum number of iterations to perform before stopping.
 * @param l2_error Reference to store the computed L2 error after convergence.
 * @param forcing_term A function that computes the forcing term (right-hand side) at given coordinates (x, y).
 * @param exact_solution A function that computes the exact solution at given coordinates (x, y) for error analysis.
 */
void Serial_Solver::Solve_Serial(Grid &grid, double tol, int max_iters, double &l2_error,
            std::function<double(double, double)> forcing_term,
            std::function<double(double, double)> exact_solution){

    Serial_Solver::initialize(grid, forcing_term);

    bool converged = false;
    int iter = 0;

    while(!converged && iter < max_iters){

        Serial_Solver::jacobi_step(grid);

        double error = Serial_Solver::compute_error(grid);

        if(error < tol){
            converged = true;
        }

        std::swap(grid.u_old, grid.u_new);

        iter++;
    }

    l2_error = Serial_Solver::compute_l2_error(grid, grid.u_old, exact_solution);

    std::cout << "n = " << grid.global_n << " Serial Solver Converged in " 
            << iter << " iterations with L2 error: " << l2_error << std::endl;

}
