#include <mariadb/conncpp.hpp>
#include <iostream>

namespace db {
    class DB {
    public:
        DB();
        std::unique_ptr<sql::Connection>  conn;

    private:
        void make_connection();
    };

}