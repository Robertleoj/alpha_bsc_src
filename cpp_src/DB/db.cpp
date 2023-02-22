#include "./db.h"
#include <torch/torch.h>
#include <istream>
#include "../config/config.h"

namespace db {

    DB::DB(std::string game){
        this->make_connection();
        this->game = game;

        this->get_game_id();
        std::cout << "Game ID: " << this->game_id << std::endl;

        this->set_curr_generation();
        std::cout << "Current generation: " << this->curr_generation << std::endl;
    }

    sqlite3_stmt * DB::query(std::string q) {
        //caller must finalize query!!
        sqlite3_stmt * stmt;

        int rc = sqlite3_prepare_v2(
            this->db, 
            q.c_str(), 
            -1, 
            &stmt,
            NULL
        );

        if(rc != SQLITE_OK){
            throw std::runtime_error(sqlite3_errmsg(this->db));
        }

        // int rc =sqlite3_step(stmt);

        // if(rc != SQLITE_OK){
        //     std::cerr << q << std::endl;
        //     throw std::runtime_error(sqlite3_errmsg(this->db));
        // }
        

        return stmt;
        
    }

    void DB::set_curr_generation(){
        std::string stmt = R"(
                select max(generation_num) as gen
                from generations
                where game_id = %d
            )";
        stmt = utils::string_format(stmt, this->game_id);


        auto res = this->query(stmt);

        int rc = sqlite3_step(res);
        if(rc == SQLITE_ROW){
            this->curr_generation = sqlite3_column_int(res, 0);
        } else {
            std::cerr << "No generation found for game id " << this->game_id << std::endl;
            throw std::runtime_error("No generation");
        }
        
        sqlite3_finalize(res);


        std::string gen_id_stmt = utils::string_format(
            R"(
                select id
                from generations
                where 
                    generation_num = %d
                    and game_id = %d
            )",
            this->curr_generation,
            this->game_id
        );

        auto q_res = this->query(gen_id_stmt);
        rc = sqlite3_step(q_res);

        if(rc == SQLITE_ROW){
            this->generation_id = sqlite3_column_int(q_res, 0);
        } else {
            std::cerr << "No generation id found for game id " << this->game_id << " and generation " << this->curr_generation << std::endl;
            throw std::runtime_error("No generation id");
        }
        sqlite3_finalize(q_res);

    }

    void DB::make_connection() {
        char *zErrMsg = 0;
        int rc;

        rc = sqlite3_open("../db/db.db", &this->db);

        if( rc ) {
            fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
            exit(0);
        } else {
            fprintf(stderr, "Opened database successfully\n");
        }

        // sqlite3_close(db);

    }

    void DB::get_game_id(){

        auto stmt = utils::string_format(
            "select id from games where game_name = \"%s\"",
            this->game.c_str()
        );

        auto res = this->query(stmt);
        int rc = sqlite3_step(res);

        if(rc == SQLITE_ROW){
            this->game_id = sqlite3_column_int(res, 0);
        } else {
            std::cerr << "No game id found for game \"" << this->game << "\"" << std::endl;

            throw std::runtime_error(sqlite3_errmsg(this->db));
        }
        sqlite3_finalize(res);
    }



    void DB::insert_training_samples(std::vector<nn::TrainingSample> &samples) {

        for(auto &sample : samples){

            sqlite3_stmt * stmt = nullptr;
            auto q = sqlite3_prepare_v2(
                this->db,
                R"(
                insert into training_data (
                    generation_id,
                    state,
                    policy,
                    outcome
                ) 
                values
                (?, ?, ?, ?)
                )",
                -1,
                &stmt,
                NULL
            );
            
            sqlite3_bind_int(stmt, 1, this->generation_id);

            std::stringstream state_ss;
            torch::save(sample.state, state_ss);
            auto state_str = state_ss.str();


            sqlite3_bind_blob(stmt, 2, state_str.c_str(), state_str.size(), SQLITE_STATIC);

            std::stringstream policy_ss;
            torch::save(sample.target_policy, policy_ss);
            auto pol_str = policy_ss.str();

            sqlite3_bind_blob(stmt, 3, pol_str.c_str(), pol_str.size(), SQLITE_STATIC);

            sqlite3_bind_double(stmt, 4, sample.outcome);

            int rc = sqlite3_step(stmt);

            if (rc != SQLITE_DONE){
                std::cerr << "Failed to insert training sample!" << std::endl;
                throw std::runtime_error("Failed to insert");
            }

            sqlite3_finalize(stmt);
        }
    }
}