#include "IO.hpp"

#include <fstream>
#include <vector>
#include <mpi.h>

/**
 * @brief Function to write the grid values to a VTK file for visualization.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param filename The base name for the output VTK files.
 * @param rank The rank of the current process
 * @param size The total number of processes 
 */
void IO:: write_vtk(
    const Grid& grid,
    const std::string& filename,
    int rank,
    int size
){
    int local_size = grid.local_rows * grid.ny; /// < Number of points in the local grid (excluding ghost cells)

    std::vector<int> recv_counts;            /// < Number of elements to receive from each process
    std::vector<int> displs;                 /// < Displacements for MPI_Gatherv

    if(rank == 0){

        recv_counts.resize(size);
        displs.resize(size);
    }

    MPI_Gather(
        &local_size,
        1,
        MPI_INT,
        recv_counts.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    std::vector<double> global_solution;

    if(rank == 0){

        int total_size = 0;

        for(int i = 0; i < size; ++i){

            displs[i] = total_size;
            total_size += recv_counts[i];
        }

        global_solution.resize(total_size);
    }

    MPI_Gatherv(
        &grid.u_old[grid.idx(1,0)],
        local_size,
        MPI_DOUBLE,

        global_solution.data(),
        recv_counts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank != 0){
        return;         /// < Only the root process will write the output file
    }

    std::ofstream out(filename);

    out << "# vtk DataFile Version 3.0\n";

    out << "Laplace Solution\n";
    
    out << "ASCII\n";

    out << "DATASET STRUCTURED_POINTS\n";

    out << "DIMENSIONS " << grid.global_n << " " << grid.global_n << " 1\n";

    out << "ORIGIN 0 0 0\n";

    out << "SPACING " << grid.h << " " << grid.h << " 1\n";

    out << "POINT DATA " << grid.global_n * grid.global_n << "\n";

    out << "SCALARS solution double\n";

    out << "LOOKUP_TABLE default\n";

    for(double v: global_solution){

        out << v << "\n";
    }

}