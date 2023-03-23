#include "./db.h"
#include <torch/torch.h>
#include <istream>
#include "../config/config.h"
#include <sys/stat.h>
#include <uuid/uuid.h>

std::vector<uint8_t> tensor_to_binary(at::Tensor t){
    std::stringstream ss;
    torch::save(t, ss);
    auto str = ss.str();
    auto buffer = str.data();
    int buffer_size = str.size();

    std::vector<uint8_t> ret(buffer_size);

    for(int i = 0; i < buffer_size; i++){
        ret[i] = buffer[i];
    }

    return ret;
}

namespace db {

    DB::DB(int generation_num){
        this->make_connection();

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
        )";

        auto res = this->query(stmt);

        int max_gen;
        int rc = sqlite3_step(res);
        if(rc == SQLITE_ROW){
            max_gen = sqlite3_column_int(res, 0);
        } else {
            std::cerr << "No generation found" << std::endl;
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
            )",
            this->curr_generation
        );

        auto q_res = this->query(gen_id_stmt);
        int rc = sqlite3_step(q_res);

        if(rc == SQLITE_ROW){
            this->generation_id = sqlite3_column_int(q_res, 0);
        } else {
            std::cerr << "No generation id found for generation " << this->curr_generation << std::endl;
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


    void DB::insert_training_samples(std::vector<TrainingSample> *samples) {
        // Write to bson file
        using json=nlohmann::json;

        json j_arr;

        for(auto &sample : *samples){
            json sample_j;

            // state
            auto state_bin = tensor_to_binary(sample.state);

            sample_j["state"] = json::binary(tensor_to_binary(sample.state));
            sample_j["policy"] = json::binary(tensor_to_binary(sample.target_policy));
            sample_j["outcome"] = sample.outcome;
            sample_j["moves"] = sample.moves;
            sample_j["player"] = (sample.player == pp::First ? 1 : 0);
            sample_j["moves_left"] = sample.moves_left;

            j_arr.push_back(sample_j);
        }

        json j_top;
        j_top["samples"] = j_arr;

        // make sure training_data directory exists
        mkdir("training_data", 0777);

        // ensure generation directory exists
        auto dir_name = utils::string_format("./training_data/%d", this->curr_generation);
        mkdir(dir_name.c_str(), 0777);

        // Create uuid
        uuid_t uuid;
        uuid_generate(uuid);

        char uuid_str[37];
        uuid_unparse(uuid, uuid_str);

        std::string fname = utils::string_format("./training_data/%d/%s.bson", this->curr_generation, uuid_str);

        auto v = json::to_bson(j_top);

        std::ofstream out(fname, std::ios::binary);
        out.write((char *)v.data(), v.size());
    }

}