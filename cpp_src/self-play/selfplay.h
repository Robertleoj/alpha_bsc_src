#pragma once

#include "../MCTS/agent.h"
#include "../DB/db.h"
#include "../NN/nn.h"
#include <string>
#include "../base/types.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <torch/all.h>
#include "../utils/utils.h"

typedef std::pair<int, at::Tensor> eval_request;


class SelfPlay {
public:
    std::unique_ptr<nn::NN> neural_net;
    std::unique_ptr<db::DB> db;
    std::string game;

    SelfPlay(std::string game);

    // void play_game();

    void self_play();

    void thread_play(
        int thread_idx, 
        utils::thread_queue<eval_request> * eval_q,
        // std::mutex * q_mutex,
        std::mutex * db_mutex,
        bool *req_completed,
        std::mutex *req_completed_mutex,
        std::unique_ptr<nn::NNOut> *evaluations,
        std::condition_variable * eval_cv,
        // std::condition_variable * nn_q_wait_cv    ,
        // std::mutex * results_mutex,
        std::atomic<int> * games_left,
        std::atomic<int> * num_active_threads
    );
};