#pragma once
#include <mariadb/conncpp.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "../NN/nn.h"

namespace db {
    class DB {
    public:
        std::string game;
        int game_id;
        DB(std::string game);
        std::unique_ptr<sql::Connection>  conn;
        int curr_generation;
        int generation_id;

        void get_game_id();

        void insert_training_samples(std::vector<nn::TrainingSample>&);

    private:
        void make_connection();
        void set_curr_generation();
    };

}