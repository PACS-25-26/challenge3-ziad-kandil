#ifndef IO_HPP
#define IO_HPP

#include "Grid.hpp"
#include "Utils.hpp"

#include <string>

class IO{

    public:

    /**
     * @brief Function to write the grid values to a VTK file for visualization.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output VTK files.
     * @param rank The rank of the current process
     * @param size The total number of processes 
     */
    static void write_vtk(
        const Grid& grid,
        const std::string& filename,
        int rank,
        int size
    );

    /**
     * @brief Function to write the solution and grid coordinates to a text file.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output text files.
     * @param rank The rank of the current process
     * @param size The total number of processes 
     */
    static void write_text(
        const Grid& grid,
        const std::string& filename,
        int rank,
        int size
    );

    /**
     * @brief Function to write the solution and grid coordinates to a csv file.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output csv files.
     * @param rank The rank of the current process
     * @param size The total number of processes 
     */
    static void write_csv(
        const Grid& grid,
        const std::string& filename,
        int rank,
        int size
    );

    /**
     * @brief Function to write the grid values to a VTK file for visualization in serial.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output VTK file.
     */
    static void write_vtk_serial(
        const Grid& grid,
        const std::string& filename
    );

    /**
     * @brief Function to write the solution and grid coordinates to a text file in serial.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output text file.
     */
    static void write_text_serial(
        const Grid& grid,
        const std::string& filename
    );

    /**
     * @brief Function to write the solution and grid coordinates to a csv file in serial.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output csv file.
     */
    static void write_csv_serial(
        const Grid& grid,
        const std::string& filename
    );
};

#endif