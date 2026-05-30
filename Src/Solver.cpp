#include "Solver.hpp"

#include <cmath>
#include <numbers>
#include <omp.h>

constexpr double pi = std::numbers::pi_v<double>;

/**
 * @brief Computes the exact solution of the problem at the given coordinates (x, y).
 * @param x The x-coordinate for which to compute the exact solution.
 * @param y The y-coordinate for which to compute the exact solution.
 * @return The computed exact solution as a double value.
 */
double Solver::exact_solution(double x, double y){
    return sin(2.0 * pi * x) * sin(2.0 * pi * y);
}

/**
 * @brief Computes the forcing term of the problem at the given coordinates (x, y).
 * @param x The x-coordinate for which to compute the forcing term.
 * @param y The y-coordinate for which to compute the forcing term.
 * @return The computed forcing term as a double value.
 */
double Solver::forcing_term(double x, double y){
    return 8.0 * pi * pi * sin(2.0 * pi * x) * sin(2.0 * pi * y);
}

/**
 * @brief Initializes the grid with the given values.
 */
void Solver::initialize(Grid& grid){

    for(int i = 1; i <= grid.local_rows; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = grid.x(j);
            double y_coord = grid.y(i);

            grid.rhs[grid.idx(i,j)] = forcing_term(x_coord, y_coord);
        }
    }
}

/**
 * @brief Performs one iteration of the Jacobi method to update the grid values.
 */
void Solver::jacobi_step(Grid& grid){

    const double h2 = grid.h * grid.h;

    #pragma omp parallel for

        for(int i = 1; i <= grid.local_rows; ++i){

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
 * @brief Computes the local error of the current grid values compared to the previous iteration.
 * Each rank will return its local sum, and the global sum will be computed in the main function using MPI_Reduce
 * @return The computed local error as a double value.
 */
double Solver::compute_local_error(const Grid& grid){

    double local_sum = 0.0;

    #pragma omp parallel for reduction (+: local_sum)

    for(int i = 1; i <= grid.local_rows; ++i){

        for(int j = 1; j < grid.ny; ++j){

            double diff = grid.u_new[grid.idx(i,j)] - grid.u_old[grid.idx(i,j)];
            local_sum += diff * diff;
        }
    }

    return local_sum;
}

/**
 * @brief Computes the local L2 error of the current grid values compared to the exact solution.
 * Each rank will return its local sum, and the global sum will be computed in the main function using MPI_Reduce
 * @return The computed local L2 error as a double value.
 */
double Solver::compute_local_l2_error(const Grid& grid, const std::vector<double>& solution){

    double local_sum = 0.0;

    #pragma omp parallel for reduction (+: local_sum)

    for(int i = 1; i <= grid.local_rows; ++i){

        for(int j = 1; j < grid.ny; ++j){

            double x_coord = grid.x(j);
            double y_coord = grid.y(i);

            double exact = exact_solution(x_coord, y_coord);
            double diff = solution[grid.idx(i,j)] - exact;
            local_sum += diff * diff;
        }
    }
    return local_sum;
}