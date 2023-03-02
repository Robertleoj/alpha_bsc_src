#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "../NN/nn.h"
#include "../sqlite/sqlite3.h"

namespace db {
    class DB {
    public:
        std::string game;
        int game_id;
        DB(std::string game);
        sqlite3 * db;
        int curr_generation;
        int generation_id;

        void get_game_id();

        void insert_training_samples(std::vector<nn::TrainingSample> *);
        ~DB();

    private:
        void make_connection();
        void set_curr_generation();
        sqlite3_stmt * query(std::string);
    };

}