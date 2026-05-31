#include "Solver.hpp"

#include <omp.h>
#include <cmath>

constexpr double pi = std::numbers::pi_v<double>;

/**
 * @brief Initializes the grid with the given values.
 * @param grid The local grid structure 
 * @param f The forcing function (right-hand side)
 */
void Solver::initialize(Grid& grid, std::function<double(double, double)> f){

    for(int i = 1; i <= grid.local_rows; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = grid.x(j);
            double y_coord = grid.y(i);

            grid.rhs[grid.idx(i,j)] = f(x_coord, y_coord);
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

    return std::sqrt(local_sum * grid.h);
}

/**
 * @brief Computes the local L2 error of the current grid values compared to the exact solution.
 * @param grid The local grid structure
 * @param solution The current solution vector to compare against the exact solution
 * @param exact_sol A function that computes the exact solution at given coordinates (x, y)
 * @return The computed local L2 error as a double value.
 */
double Solver::compute_local_l2_error(const Grid& grid, const std::vector<double>& solution, std::function<double(double, double)> exact_sol){

    double local_sum = 0.0;

    #pragma omp parallel for reduction (+: local_sum)

    for(int i = 1; i <= grid.local_rows; ++i){

        for(int j = 1; j < grid.ny; ++j){

            double x_coord = grid.x(j);
            double y_coord = grid.y(i);

            double exact = exact_sol(x_coord, y_coord);
            double diff = solution[grid.idx(i,j)] - exact;
            local_sum += diff * diff;
        }
    }
    return local_sum;
}