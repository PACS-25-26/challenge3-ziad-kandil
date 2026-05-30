#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include "Grid.hpp"

class Parallel{

    public:

    /**
     * @brief Function to exchange ghost rows between neighboring processes using MPI.
     * Each process will send its top row to the process above and
     * receive the bottom ghost row from the process below, and vice versa.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param rank The rank of the current process.
     * @param size The total number of processes in the MPI communicator.
     */
    static void exchange_ghost_rows(Grid& grid, int rank, int size);

}

#endif