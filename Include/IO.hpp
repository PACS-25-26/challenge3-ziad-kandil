#ifndef IO_HPP
#define IO_HPP

#include "Grid.hpp"

#include <string>

class IO{

    public:

    /**
     * @brief Function to write the grid values to a VTK file for visualization.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The base name for the output VTK files.
     * @param rank The rank of the current process
     * @param size The total number of processes 
     */
    static void write_vtk(
        const Grid& grid,
        const std::string& filename,
        int rank,
        int size
    );
};

#endif