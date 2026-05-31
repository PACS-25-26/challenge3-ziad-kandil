#include "Grid.hpp"
#include "Solver.hpp"
#include "Parallel.hpp"
#include "Utils.hpp"
#include "IO.hpp"

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>


int main(int argc, char** argv){

    MPI_Init(&argc, &argv);
    
    omp_set_num_threads(2);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 64;
    int max_iters = 10000;
    double tol = 1e-6;

    int local_rows;
    int start_row;

    auto forcing_term = [](double x, double y){
        return 8.0 * M_PI * M_PI * 
                std::sin(2.0 * M_PI * x) * 
                std::sin(2.0 * M_PI * y);
    };

    auto exact_solution = [](double x, double y){
        return std::sin(2.0 * M_PI * x) * 
                std::sin(2.0 * M_PI * y);
    };

    Utils::compute_partitions(n, size, rank, local_rows, start_row);
    Grid grid(n, local_rows, start_row);

    Solver::initialize(grid, forcing_term);

    bool converged = false;
    int iter = 0;

    while(!converged && iter < max_iters){

        Parallel::exchange_ghost_rows(grid, rank, size);

        Solver::jacobi_step(grid);

        double local_error = Solver::compute_local_error(grid);

        bool local_converged = (local_error < tol);

        int local_flag = local_converged ? 1 : 0;
        int global_flag = 0;

        MPI_Allreduce(
            &local_flag,
            &global_flag,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD
        );

        converged = (global_flag == 1);

        std::swap(grid.u_old, grid.u_new);

        ++iter;
    }


    double local_l2_error = Solver::compute_local_l2_error(
        grid,
        grid.u_old,
        exact_solution
    );

    double global_l2_error;

    MPI_Allreduce(
        &local_l2_error,
        &global_l2_error,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    global_l2_error = std::sqrt(grid.h * global_l2_error);

    if(rank == 0){
        std::cout << "Converged in " 
            << iter << " iterations with L2 error: " << global_l2_error << std::endl;
    }

    IO::write_vtk(grid, "solution.vtk", rank, size);
    IO::write_text(grid, "solution.txt", rank, size);
    IO::write_csv(grid, "solution.csv", rank, size);

    MPI_Finalize();
    return 0;

}