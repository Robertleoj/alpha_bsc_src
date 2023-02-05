#include "./db.h"

namespace db {

    DB::DB(){
        this->make_connection();
    }

    void DB::make_connection() {

        // Instantiate Driver
        sql::Driver* driver = sql::mariadb::get_driver_instance();

        sql::SQLString url("jdbc:mariadb://localhost:3306/self_play");

        sql::SQLString user("user");
        sql::SQLString pw("password");

        // Establish Connection
        // Use a smart pointer for extra safety
        std::unique_ptr<sql::Connection> conn(driver->connect(url, user, pw));

        this->conn = std::move(conn);

        std::cout << "Connection valid: " << this->conn->isValid() << std::endl;;
        std::cout << "Schema: " << this->conn->getSchema() << std::endl;;

        // if (!conn) {
        //     std::cerr << "Invalid database connection" << std::endl;
        //     exit (EXIT_FAILURE);
        // }
    }
}