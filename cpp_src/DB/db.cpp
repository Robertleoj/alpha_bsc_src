#include "./db.h"
#include <stdexcept>
#include <torch/torch.h>
#include <istream>
#include "../config/config.h"
#include <sys/stat.h>

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




    void DB::insert_evaluations(std::vector<EvalEntry> & eval_entries){
        
        using json = nlohmann::json;

        json j_arr;

        for(auto & eval_entry : eval_entries){
            json j;
            // j["generation_id"] = this->generation_id;
            j["moves"] = eval_entry.moves;
            j["search_depth"] = eval_entry.search_depth;
            j["policy_target"] = eval_entry.policy_target;
            j["value_target"] = eval_entry.value_target;
            j["policy_prior"] = eval_entry.policy_prior;
            j["policy_mcts"] = eval_entry.policy_mcts;
            j["nn_value"] = eval_entry.nn_value;
            j["nn_value_error"] = eval_entry.nn_value_error;
            j["mcts_value"] = eval_entry.mcts_value;
            j["mcts_value_error"] = eval_entry.mcts_value_error;
            j["policy_prior_error"] = eval_entry.policy_prior_error;
            j["policy_mcts_error"] = eval_entry.policy_mcts_error;

            j_arr.push_back(j);
        }

        json j_top;
        j_top["evals"] = j_arr;

        std::string json_str = j_top.dump(4);

        std::string folder = "evals";
        mkdir(folder.c_str(), 0777);
        std::string file_name = folder + "/" + std::to_string(this->curr_generation) + ".json";
        std::ofstream file(file_name);
        file << json_str;
        file.close();
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
            throw std::runtime_error("Can't open database");
        } else {
            fprintf(stdout, "Opened database successfully\n");
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
            sample_j["weight"] = sample.weight;

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
        std::string uuid_str = utils::make_uuid();

        std::string fname = utils::string_format("./training_data/%d/%s.bson", this->curr_generation, uuid_str.c_str());

        auto v = json::to_bson(j_top);

        std::ofstream out(fname, std::ios::binary);
        out.write((char *)v.data(), v.size());
    }

}