#include "Parallel.hpp"

#include <mpi.h>

/**
 * @brief Function to exchange ghost rows between neighboring processes using MPI.
 * Each process will send its top row to the process above and
 * receive the bottom ghost row, and vice versa.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param rank The rank of the current process.
 * @param size The total number of processes in the MPI communicator.
 */
void Parallel::exchange_ghost_rows(Grid& grid, int rank, int size){

    int up = rank - 1;      /// < Rank of the process above
    int down = rank + 1;    /// < Rank of the process below

    if(up >= 0){

        /// < Send top row to the process above
        /// < Receive top ghost row from upward neighbor

        MPI_Sendrecv(
            &grid.u_old[grid.idx(1, 0)],
            grid.ny,
            MPI_DOUBLE,
            up,
            0,

            &grid.u_old[grid.idx(0, 0)],
            grid.ny,
            MPI_DOUBLE,
            up,
            1,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    if(down < size){

        /// < Send bottom row to the process below
        /// < Receive bottom ghost row from downward neighbor

        MPI_Sendrecv(
            &grid.u_old[grid.idx(grid.local_rows, 0)],
            grid.ny,
            MPI_DOUBLE,
            down,
            1,

            &grid.u_old[grid.idx(grid.local_rows + 1, 0)],
            grid.ny,
            MPI_DOUBLE,
            down,
            0,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }
}