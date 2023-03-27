#include "Solver.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <thread>


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

double score_to_outcome(int score){
    if(score == -1000){
        return -2;
    } else if (score < 0){
        return -1;
    } else if (score > 0){
        return 1;
    } else {
        return 0;
    }
}

double scores_to_eval(std::vector<int> scores){

    double eval = -1000;

    for(auto s: scores){
        eval = std::max(eval, score_to_outcome(s));
    }

    return eval;
}

void evaluate_chunk(std::vector<std::string>* positions, std::string book_path, std::vector<int> *evaluations){
    using namespace GameSolver::Connect4;
    Solver solver;

    solver.loadBook(book_path);
    
    for(auto &s: *positions){
        Position P;

        if(P.play(s) != s.size()) {
            throw std::runtime_error("Invalid move sequence");
        } else {
            std::vector<int> scores = solver.analyze(P, false);
            evaluations->push_back(scores_to_eval(scores));
        }
    }
}


std::vector<int> evaluate_many(std::vector<std::string>& positions, std::string book_path){
    using namespace GameSolver::Connect4;
    int THREADS = std::thread::hardware_concurrency();

    std::vector<int> evaluations[THREADS];

    std::vector<int> all_evaluations;

    std::vector<std::thread> threads;

    std::vector<std::string> chunks[THREADS];

    int positions_per_thread = positions.size() / THREADS;

    // chunk the positions into THREAD chunks
    for(int i = 0; i < positions.size(); i++){

        int idx = i / positions_per_thread;

        if(idx >= THREADS){
            idx = THREADS - 1;
        }

        chunks[idx].push_back(positions[i]);
    }

    for(int i = 0; i < THREADS; i++){
        threads.push_back(std::thread(
            evaluate_chunk, 
            &chunks[i], 
            book_path, 
            &evaluations[i]
        ));
    }

    for(int i = 0; i < THREADS; i++){
        threads[i].join();
    }

    for(int i = 0; i < THREADS; i++){
        for(auto v: evaluations[i]){
            all_evaluations.push_back(v);
        }
    }

    return all_evaluations;
}

namespace py = pybind11;

PYBIND11_MODULE(conn4_solver, m) {
    // m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("solve", &solve, "Solve connect4 position");
    m.def("evaluate_many", &evaluate_many, "evaluate many positions");
}


