#include "Solver.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


/**
 * Main function.
 * Reads Connect 4 positions, line by line, from standard input
 * and writes one line per position to standard output containing:
 *  - score of the position
 *  - number of nodes explored
 *  - time spent in microsecond to solve the position.
 *
 *  Any invalid position (invalid sequence of move, or already won game)
 *  will generate an error message to standard error and an empty line to standard output.
 */
std::vector<int> solve(std::vector<std::string>& moves) {
    using namespace GameSolver::Connect4;
    Solver solver;

    std::string opening_book = "7x6.book";
    solver.loadBook(opening_book);


    std::stringstream ss;
    for(auto move : moves) {
        ss << move;
    }

    std::string move_str = ss.str();

    Position P;

    if(P.play(move_str) != move_str.size()) {
        throw std::runtime_error("Invalid move sequence");
    } else {
        std::vector<int> scores = solver.analyze(P, false);
        return scores;
    }
}

namespace py = pybind11;

PYBIND11_MODULE(conn4_solver, m) {
    // m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("solve", &solve, "Solve connect4 position");
}


