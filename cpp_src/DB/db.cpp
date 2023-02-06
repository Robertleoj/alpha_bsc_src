#include "./db.h"
#include <torch/torch.h>
#include <istream>

namespace db {

    DB::DB(std::string game){
        this->make_connection();
        this->game = game;

        this->get_game_id();
        std::cout << "Game ID: " << this->game_id << std::endl;

        this->set_curr_generation();
        std::cout << "Current generation: " << this->curr_generation << std::endl;
    }

    void DB::set_curr_generation(){
        auto stmt = this->conn->prepareStatement(R"(
                select max(generation_num) as gen
                from generations
                where game_id = ?
            )");
        stmt->setInt(1, this->game_id);
        auto res = stmt->executeQuery();
        
        if(res->next()){
            this->curr_generation = res->getInt("gen");
        } else {
            std::cerr << "No generation found for game id " << this->game_id << std::endl;
            throw std::runtime_error("No generation");
        }

        auto gen_id_stmt = this->conn->prepareStatement(R"(
            select id
            from generations
            where 
                generation_num = ?
                and game_id = ? 
        )");

        gen_id_stmt->setInt(1, this->curr_generation);
        gen_id_stmt->setInt(2, this->game_id);

        auto gen_id_res = gen_id_stmt->executeQuery();
        if(gen_id_res->next()){
            this->generation_id = gen_id_res->getInt("id");
        } else {
            std::cerr << "No generation id found for game id " << this->game_id << " and generation " << this->curr_generation << std::endl;
            throw std::runtime_error("No generation id");
        }

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

        // this does not work
        if (this->conn->isValid()) {
            std::cout << "DB connection valid" << std::endl;
        } else {
            throw std::runtime_error("SQL connection invalid");
        }
    }

    void DB::get_game_id(){

        auto stmt = this->conn->prepareStatement(
            "select id from games where game_name = ?"
        );

        stmt->setString(1, this->game);

        auto res = stmt->executeQuery();

        if(res->next()){
            this->game_id = res->getInt("id");
        } else {
            std::cerr << "No game id found for game \"" << this->game << "\"" << std::endl;
            throw std::runtime_error("");
        }
    }



    void DB::insert_training_samples(std::vector<nn::TrainingSample> &samples) {

        for(auto &sample : samples){

            auto stmt = this->conn->prepareStatement(R"(
                insert into training_data (
                    generation_id,
                    state,
                    policy,
                    outcome
                ) 
                values
                (?, ?, ?, ?)
            )");

            std::stringstream state_ss;
            torch::save(sample.state, state_ss);

            std::stringstream policy_ss;
            torch::save(sample.target_policy, policy_ss);

            stmt->setInt(1, this->generation_id);

            stmt->setBlob(2, &state_ss);
            stmt->setBlob(3, &policy_ss);
            stmt->setFloat(4, sample.outcome);

            stmt->executeUpdate();
            delete stmt;
        }
    }
}