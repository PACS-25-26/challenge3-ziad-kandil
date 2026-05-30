#ifndef UTILS_HPP
#define UTILS_HPP

class Utils{

    public:

    /**
     * @brief Function to compute the local number of rows
     *  and the starting row index for each process
     */
    static void compute_partitions(
        int global_n,
        int size,
        int rank,
        int& local_rows,
        int& start_row
    )
}

#endif