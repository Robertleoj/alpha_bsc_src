#pragma once
#include <mariadb/conncpp.hpp>
#include <iostream>
#include <string>

namespace db {
    class DB {
    public:
        std::string game;
        int game_id;
        DB(std::string game);
        std::unique_ptr<sql::Connection>  conn;
        int curr_generation;

        void get_game_id();

    private:
        void make_connection();
        void set_curr_generation();
    };

}