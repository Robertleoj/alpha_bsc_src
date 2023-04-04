#pragma once

#include <ATen/core/TensorBody.h>
#include "../NN/nn.h"
#include "../config/config.h"
#include "../DB/db.h"
#include "../MCTS/agent.h"


struct EvalRequest {
    bool completed;
    at::Tensor state;
    std::unique_ptr<nn::NNOut> result;
    std::vector<game::move_id> * legal_moves;
};

struct Batch {
    std::vector<EvalRequest*> requests;
    at::Tensor batch_tensor;
    std::pair<at::Tensor, at::Tensor> result;
    std::vector<std::vector<game::move_id> *> legal_moves;
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
