#pragma once

#include "../DB/db.h"
#include "../MCTS/agent.h"
#include "../NN/nn.h"
#include "../base/types.h"
#include "../utils/utils.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <torch/all.h>


namespace sim {

    void self_play(std::string game);

    // no game required as we only do this for connect4
    void eval_targets(std::string, int generation_num = -1);

}



// //         // variables
//     std::unique_ptr<nn::NN> neural_net;
//     std::unique_ptr<db::DB> db;
//     std::string game;

//     void thread_play(
//         int thread_idx, 
//         ThreadData *thread_data
//     );

//     eval_f make_eval_function(
//         int thread_idx, 
//         ThreadData *thread_data, 
//         nn::NN *neural_net
//     );

//     void start_threads(
//         std::thread *threads, 
//         ThreadData *thread_data, 
//         int num_threads
//     );

//     void thread_game(
//         int thread_idx, 
//         ThreadData *thread_data
//     );

//     game::IGame *get_game_instance();

//     void write_samples(
//         std::vector<nn::TrainingSample> *, 
//         Agent *, 
//         ThreadData *,
//         game::IGame *
//     );

//     nn::TrainingSample get_training_sample(
//         nn::move_dist visit_counts,
//         std::string moves, Board
//     );

//     game::move_id select_move(
//         nn::move_dist, 
//         int num_moves
//     );

//     void pop_batch(
//         ThreadData &thread_data, 
//         std::function<int()> bs,
//         std::vector<int> &thread_indices,
//         std::vector<at::Tensor> &states
//     );

//     void eval_batch(
//         std::vector<at::Tensor> &states,
//         std::vector<int> &thread_indices, 
//         ThreadData &thread_data
//     );

//     void active_thread_work(
//         ThreadData &thread_data,
//         std::function<unsigned long()> bs
//     );

