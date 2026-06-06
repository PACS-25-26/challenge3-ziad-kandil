#include "Grid.hpp"
#include "Parallel_Solver.hpp"
#include "Serial_Solver.hpp"
#include "Utils.hpp"
#include "IO.hpp"

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>


int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto forcing_term = [](double x, double y){
        return 8.0 * M_PI * M_PI * 
                std::sin(2.0 * M_PI * x) * 
                std::sin(2.0 * M_PI * y);
    };

    auto exact_solution = [](double x, double y){
        return std::sin(2.0 * M_PI * x) * 
                std::sin(2.0 * M_PI * y);
    };

    std::vector<int> grid_sizes = {16, 32, 64, 128, 256};
    int max_iters = 50000;
    double tol = 1e-9;

    int n_threads = 2;
    omp_set_num_threads(n_threads);

    int local_rows;
    int start_row;

    std::vector<double> Parallel_timings;
    std::vector<double> Serial_timings;
    std::vector<double> Parallel_l2_errors;
    std::vector<double> Serial_l2_errors;

    Parallel_timings.reserve(grid_sizes.size());
    Serial_timings.reserve(grid_sizes.size());
    Parallel_l2_errors.reserve(grid_sizes.size());
    Serial_l2_errors.reserve(grid_sizes.size());

    for (int n : grid_sizes){

        Utils::compute_partitions(n, size, rank, local_rows, start_row);

        Grid Parallel_grid(n, local_rows, start_row);

        double Parallel_l2_error = 0.0;

        double start_time = MPI_Wtime();

        Parallel_Solver::Solve_Parallel(Parallel_grid, rank, size, tol, max_iters, Parallel_l2_error, forcing_term, exact_solution);

        double end_time = MPI_Wtime();
        double Parallel_time = end_time - start_time;

        Parallel_timings.push_back(Parallel_time);
        Parallel_l2_errors.push_back(Parallel_l2_error);

        std::string vtk_output_filename = "Parallel_solution_" + std::to_string(Parallel_grid.global_n) + ".vtk";
        std::string text_output_filename = "Parallel_solution_" + std::to_string(Parallel_grid.global_n) + ".txt";
        std::string csv_output_filename = "Parallel_solution_" + std::to_string(Parallel_grid.global_n) + ".csv";

        IO::write_vtk(Parallel_grid, vtk_output_filename, rank, size);
        IO::write_text(Parallel_grid, text_output_filename, rank, size);
        IO::write_csv(Parallel_grid, csv_output_filename, rank, size);

        if (rank ==0){

            Grid Serial_grid(n, n, 0);

            double Serial_l2_error = 0.0;

            double Serial_start_time = MPI_Wtime();

            Serial_Solver::Solve_Serial(Serial_grid, tol, max_iters, Serial_l2_error, forcing_term, exact_solution);

            double Serial_end_time = MPI_Wtime();
            double Serial_time = Serial_end_time - Serial_start_time;

            Serial_timings.push_back(Serial_time);
            Serial_l2_errors.push_back(Serial_l2_error);

            std::string serial_vtk_output_filename = "Serial_solution_" + std::to_string(Serial_grid.global_n) + ".vtk";
            std::string serial_text_output_filename = "Serial_solution_" + std::to_string(Serial_grid.global_n) + ".txt";
            std::string serial_csv_output_filename = "Serial_solution_" + std::to_string(Serial_grid.global_n) + ".csv";
            IO::write_vtk_serial(Serial_grid, serial_vtk_output_filename);
            IO::write_text_serial(Serial_grid, serial_text_output_filename);
            IO::write_csv_serial(Serial_grid, serial_csv_output_filename);
        }

    }

    if(rank == 0){

        IO::save_data(Parallel_timings, Serial_timings, grid_sizes, "timings.dat");
        IO::save_data(Parallel_l2_errors, Serial_l2_errors, grid_sizes, "l2_errors.dat");
        
        int status_1 =system(
            "gnuplot -persist -e \""
            "set title 'Timings Results';"
            "set ylabel 'Time [s]';"
            "set xlabel 'Number of Grid Points n';"
            "set grid;"
            "set key left top;"
            "plot 'Test/data/timings.dat' u 1:2 w lp title 'Parallel',"
            "     'Test/data/timings.dat' u 1:3 w lp title 'Serial'"
            "\""
        );

        int status_2 = system(
            "gnuplot -persist -e \""
            "set title 'L2 Errors Results';"
            "set ylabel 'Error';"
            "set xlabel 'Number of Grid Points n';"
            "set grid;"
            "plot 'Test/data/l2_errors.dat' u 1:2 w lp title 'Parallel',"
            "     'Test/data/l2_errors.dat' u 1:3 w lp title 'Serial'"
            "\""
        );

    }

    MPI_Finalize();

    return 0;
}