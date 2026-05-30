#include "Grid.hpp"
#include "Solver.hpp"
#include "Parallel.hpp"
#include "Utils.hpp"
#include "IO.hpp"

#include <mpi.h>
#include <iostream>
#include <cmath>

int main(int argc, char** argv[]){

    MPI_init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 128;
    int max_iters = 10000;
    double tol = 1e-6;

    int local_rows;
    int start_row;

    Utils::compute_partitions(n, size, rank, local_rows, start_row);

    Grid grid(n, local_rows, start_row);

    Solver::initialize(grid);

    double global_error = 1.0;

    int iter = 0;

    while(global_error > tol && iter < max_iters){

        Parallel::exchange_ghost_rows(grid, rank, size);

        Solver::jacobian_step(grid);

        double local_error = Solver::compute_local_error(grid);

        MPI_Allreduce(
            &local_error,
            &global_error,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        )

        global_error = std::sqrt(grid.h * global_error);

        std::swap(grid.u_old, grid.u_new);

        ++iter;
    }

    double local_l2_error = Solver::compute_local_l2_error(
        grid,
        grid.u_old
    );

    double global_l2_error;

    MPI_Allreduce(
        &local_l2_error,
        &global_l2_error,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    )

    global_l2_error = std::sqrt(grid.h * global_l2_error);

    if(rank == 0){
        std::cout << "Converged in " 
            << iter << " iterations with L2 error: " << global_l2_error << std::endl;
    }

    IO::write_vtk(grid, "solution.vtk", rank, size);

    MPI_Finalize();

    return 0;

}