#include "./db.h"
#include <torch/torch.h>
#include <istream>
#include "../config/config.h"

namespace db {

    DB::DB(std::string game, int generation_num){
        this->make_connection();
        this->game = game;

        this->get_game_id();
        std::cout << "Game ID: " << this->game_id << std::endl;

        this->set_curr_generation(generation_num);
        std::cout << "Current generation: " << this->curr_generation << std::endl;
    }

    DB::~DB(){
        sqlite3_close(this->db);
    }



    void DB::insert_evaluation(EvalEntry * eval_entry){

        sqlite3_exec(this->db, "BEGIN TRANSACTION", NULL, NULL, NULL);


        sqlite3_stmt * stmt = nullptr;

        sqlite3_prepare_v2(
            this->db,
            R"(
            insert into ground_truth_evals (
                generation_id,
                moves,
                search_depth,
                policy_target,
                value_target,
                policy_prior,
                policy_mcts,
                nn_value,
                nn_value_error,
                mcts_value,
                mcts_value_error,
                prior_error,
                mcts_error
            ) 
            values
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            )",
            -1,
            &stmt,
            NULL
        );
        

        // generation_id
        sqlite3_bind_int(stmt, 1, this->generation_id);

        // moves
        sqlite3_bind_text(
            stmt, 2, eval_entry->moves.c_str(), eval_entry->moves.length(), SQLITE_TRANSIENT
        );

        // search_depth
        sqlite3_bind_int(stmt, 3, eval_entry->search_depth);

        // policy_target
        std::string policy_target = nlohmann::json(eval_entry->policy_target).dump();
        sqlite3_bind_text(stmt, 4, policy_target.c_str(), policy_target.length(), SQLITE_TRANSIENT);

        // value_target
        sqlite3_bind_double(stmt, 5, eval_entry->value_target);

        // policy_prior
        std::string policy_prior = nlohmann::json(eval_entry->policy_prior).dump();
        sqlite3_bind_text(stmt, 6, policy_prior.c_str(), policy_prior.length(), SQLITE_TRANSIENT);

        // policy_mcts
        std::string policy_mcts = nlohmann::json(eval_entry->policy_mcts).dump();
        sqlite3_bind_text(stmt, 7, policy_mcts.c_str(), policy_mcts.length(), SQLITE_TRANSIENT);

        // nn_value
        sqlite3_bind_double(stmt, 8, eval_entry->nn_value);

        // nn_value_error
        sqlite3_bind_double(stmt, 9, eval_entry->nn_value_error);
        // std::cout << "nn_value_error: " << eval_entry->nn_value_error << std::endl;

        // mcts_value
        sqlite3_bind_double(stmt, 10, eval_entry->mcts_value);

        // mcts_value_error
        sqlite3_bind_double(stmt, 11, eval_entry->mcts_value_error);

        // prior_error
        sqlite3_bind_double(stmt, 12, eval_entry->policy_prior_error);

        // mcts_error
        sqlite3_bind_double(stmt, 13, eval_entry->policy_mcts_error);

        int rc = sqlite3_step(stmt);

        if (rc != SQLITE_DONE){
            std::cerr << "Failed to insert evaluation! " << rc << std::endl;
            std::cerr << sqlite3_errmsg(this->db) << std::endl;
            throw std::runtime_error("Failed to insert");
        }

        sqlite3_finalize(stmt);
        sqlite3_exec(this->db, "COMMIT TRANSACTION", NULL, NULL, NULL);
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

        return stmt;
    }

    int DB::get_max_generation(){
        std::string stmt = R"(
            select max(generation_num) as gen
            from generations
            where game_id = %d
        )";

        stmt = utils::string_format(stmt, this->game_id);


        auto res = this->query(stmt);

        int max_gen;
        int rc = sqlite3_step(res);
        if(rc == SQLITE_ROW){
            max_gen = sqlite3_column_int(res, 0);
        } else {
            std::cerr << "No generation found for game id " << this->game_id << std::endl;
            throw std::runtime_error("No generation");
        }
        
        sqlite3_finalize(res);

        return max_gen;
    }

    void DB::set_curr_generation(int generation_num){
        if(generation_num == -1){
            this->curr_generation = this->get_max_generation();
        } else {
            this->curr_generation = generation_num;
        }


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
        int rc = sqlite3_step(q_res);

        if(rc == SQLITE_ROW){
            this->generation_id = sqlite3_column_int(q_res, 0);
        } else {
            std::cerr << "No generation id found for game id " << this->game_id << " and generation " << this->curr_generation << std::endl;
            throw std::runtime_error("No generation id");
        }
        sqlite3_finalize(q_res);

    }

    void DB::make_connection() {
        int rc;

        rc = sqlite3_open("./db.db", &this->db);

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



    void DB::insert_training_samples(std::vector<TrainingSample> *samples) {
        sqlite3_exec(this->db, "BEGIN TRANSACTION", NULL, NULL, NULL);
        for(auto &sample : *samples){

            sqlite3_stmt * stmt = nullptr;
            sqlite3_prepare_v2(
                this->db,
                R"(
                insert into training_data (
                    generation_id,
                    state,
                    policy,
                    outcome,
                    moves,
                    player,
                    moves_left
                ) 
                values
                (?, ?, ?, ?, ?, ?, ?)
                )",
                -1,
                &stmt,
                NULL
            );
            
            // generation_id
            sqlite3_bind_int(stmt, 1, this->generation_id);

            // state
            std::stringstream state_ss;
            torch::save(sample.state, state_ss);
            auto state_str = state_ss.str();

            sqlite3_bind_blob(stmt, 2, state_str.c_str(), state_str.size(), SQLITE_STATIC);

            // target policy
            std::stringstream policy_ss;
            torch::save(sample.target_policy, policy_ss);
            auto pol_str = policy_ss.str();

            sqlite3_bind_blob(stmt, 3, pol_str.c_str(), pol_str.size(), SQLITE_STATIC);

            // outcome
            sqlite3_bind_double(stmt, 4, sample.outcome);

            // moves
            sqlite3_bind_text(stmt, 5, sample.moves.c_str(), sample.moves.size(), SQLITE_TRANSIENT);

            // player
            sqlite3_bind_int(stmt, 6, (sample.player == pp::First ? 1 : 0));

            // moves_left
            sqlite3_bind_int(stmt, 7, sample.moves_left);

            int rc = sqlite3_step(stmt);

            if (rc != SQLITE_DONE){
                std::cerr << "Failed to insert training sample!" << std::endl;
                std::cerr << sqlite3_errmsg(this->db) << std::endl;
                throw std::runtime_error("Failed to insert");
            }

            sqlite3_finalize(stmt);
        }
        sqlite3_exec(this->db, "COMMIT TRANSACTION", NULL, NULL, NULL);
    }
}