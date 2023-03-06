#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "../NN/nn.h"
#include "../sqlite/sqlite3.h"

namespace db {
    struct EvalEntry {
        std::string moves;
        int search_depth;

        std::vector<double> policy_target;
        double value_target;

        std::vector<double> policy_prior;
        double policy_prior_error;

        std::vector<double> policy_mcts;
        double policy_mcts_error;

        double nn_value;
        double nn_value_error;

        double mcts_value;
        double mcts_value_error;
    };

    class DB {
    public:
        std::string game;
        int game_id;
        DB(std::string game, int generation_num = -1);
        sqlite3 * db;
        int curr_generation;
        int generation_id;

        void get_game_id();

        void insert_training_samples(std::vector<nn::TrainingSample> *);
        void insert_evaluation(EvalEntry *);
        ~DB();

    private:
        void make_connection();
        int get_max_generation();
        void set_curr_generation(int generation_num = -1);
        sqlite3_stmt * query(std::string);
    };

}