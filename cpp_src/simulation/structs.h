#pragma once

#include <ATen/core/TensorBody.h>
#include "../NN/nn.h"
#include "../config/config.h"
#include "../DB/db.h"
#include "../MCTS/agent.h"
#include <cmath>
#include <math.h>


struct EvalRequest {
    bool completed;
    at::Tensor state;
    std::unique_ptr<nn::NNOut> result;
    std::vector<game::move_id>* legal_moves;
    pp::Player to_move;
};

struct Batch {
    std::vector<EvalRequest*> requests;
    at::Tensor batch_tensor;
    std::pair<at::Tensor, at::Tensor> result;
    std::vector<std::vector<game::move_id>*> legal_moves;
    std::vector<pp::Player> to_move;
};


struct BatchData {
    std::queue<Batch> batch_queue;
    std::mutex batch_queue_mutex;
    std::condition_variable batch_queue_cv;
    std::queue<Batch> batch_result_queue;
    std::mutex batch_result_queue_mutex;
    std::condition_variable batch_result_queue_cv;
};

struct ThreadData {
    std::queue<EvalRequest*> eval_q;
    std::mutex q_mutex;
    std::mutex db_mutex;
    std::mutex start_game_mutex;
    std::condition_variable q_cv;
    std::atomic<int> games_left;
    std::atomic<int> num_active_games;
    db::DB* db;
    nn::NN* neural_net;

    ThreadData(
        nn::NN* neural_net,
        db::DB* db,
        int num_games
    ):
        q_mutex(),
        db_mutex(),
        start_game_mutex(),
        q_cv(),
        db(db),
        neural_net(neural_net) {
        // set all variables
        this->games_left = num_games;
        this->num_active_games = config::hp["self_play_num_threads"].get<int>() * config::hp["games_per_thread"].get<int>();
        this->eval_q = std::queue<EvalRequest*>();
    }

    ~ThreadData() {}
};

struct GroundTruthRequest {
    std::string moves;
    std::vector<double> ground_truth;
    double value;
};

struct EvalData {
    std::queue<GroundTruthRequest> board_queue;
    std::mutex board_queue_mutex;
    std::vector<db::EvalEntry> eval_entries;
};

struct ThreadEvalData {
    Agent* agent = nullptr;
    game::IGame* game = nullptr;
    EvalRequest request;
    GroundTruthRequest gt_request;
    bool dead_game;
    nn::NNOut first_nn_out;
    bool first_nn_out_set;

    ThreadEvalData() {
        dead_game = false;
        first_nn_out_set = false;
    }
};


struct EndgamePlayoutWeights {
    double shift;
    double slope;

    EndgamePlayoutWeights(double generation) {
        double uniform_at =
            config::hp["endgame_playout_uniform_generation"].get<double>();
        double uni_const =
            config::hp["endgame_playout_uniform_const"].get<double>();
        double base_shift = config::hp["endgame_playout_shift"].get<double>();
        this->slope = config::hp["endgame_playout_slope"].get<double>();
        double power = config::hp["endgame_playout_power"].get<double>();

        double b = config::hp["endgame_playout_b"].get<double>();

        double p = -std::log(b * std::pow(generation / uniform_at, power) + 1) / std::log(b);
        this->shift = base_shift * (1 - p) + (uni_const)*p;
    }

    double operator()(double x) { return 1 / (1 + exp(shift - x / slope)); }
};

struct MonotoneIncreasingWeights {
    double w;
    MonotoneIncreasingWeights(double generation) {
        double min = config::hp["monotone_min_w"].get<double>();
        double max = 1.0;
        double uniat = config::hp["monotone_uniform_generation"].get<double>();
        // linaerly interpolate between min and max
        this->w = min + (max - min) * generation / uniat;
        // clamp to [min, max]
        this->w = std::max(min, std::min(max, this->w));
    }

    double operator()() {
        return this->w;
    }
};

struct RandomizedCapWeights {
    double n;
    double N;
    double search_depth;
    double p;
    bool valid = true;

    RandomizedCapWeights(double generation) {

        if(
            !config::has_key("use_randomized_cap") 
            || !config::hp["use_randomized_cap"].get<bool>()
        ){
            this->valid = false;
            return;
        }

        double low_min = config::hp["randomized_cap_n_min"].get<double>();
        double low_max = config::hp["randomized_cap_n_max"].get<double>();
        double high_min = config::hp["randomized_cap_N_min"].get<double>();
        this->search_depth = config::hp["search_depth"].get<double>();
        double uniat = config::hp["randomized_cap_uniform_generation"].get<double>();

        this->p = config::hp["randomized_cap_p"].get<double>();

        this->n = low_min + (low_max - low_min) * generation / uniat;
        // clamp to [low_min, low_max]
        this->n = std::max(low_min, std::min(low_max, this->n));

        this->N = high_min + (search_depth - high_min) * generation / uniat;
        // clamp to [high_min, high_max]
        this->N = std::max(high_min, std::min(search_depth, this->N));
    }

    double operator()() {
        if(!this->valid){
            throw std::runtime_error("not using randomized cap, but trying to access it");
        }
        double playouts;
        double mult = 1;
        double r = (double) rand() / RAND_MAX;

        if (r < p) {
            playouts = this->N;
        } else {
            playouts = this->n;
            mult = -1;
        }

        return mult * (playouts / search_depth);
    }
};

