#pragma once

#include "structs.h"
#include <vector>
#include <thread>
#include <string>
#include "../NN/connect4_nn.h"
#include "../NN/breakthrough_nn.h"
#include "../games/game.h"
#include "../games/connect4.h"
#include "../games/breakthrough.h"


void join_threads(std::vector<std::vector<std::thread>*> threads);

nn::NN* get_neural_net(std::string game, db::DB* db);

ThreadData* init_thread_data(std::string game_name, int num_games, int generation_num=-1);

void pop_batch(
    ThreadData* thread_data,
    std::function<int()> bs,
    std::vector<EvalRequest*>* states
);

void dl_thread_work(std::queue<Batch>* batch_queue, ThreadData* thread_data, std::mutex* batch_queue_mutex, std::condition_variable* batch_queue_cv);

void nn_thread_work(
    std::queue<Batch>* batch_queue,
    std::queue<Batch>* batch_result_queue,
    ThreadData* thread_data,
    std::mutex* batch_queue_mutex,
    std::condition_variable* batch_queue_cv,
    std::mutex* batch_result_queue_mutex,
    std::condition_variable* batch_result_queue_cv
);

void eval_batch(Batch* batch, ThreadData* thread_data);

void return_thread_work(ThreadData* thread_data, std::queue<Batch>* batch_res_queue, std::mutex* batch_res_queue_mutex, std::condition_variable* batch_res_queue_cv);

void start_batching_threads(
    ThreadData* thread_data,
    BatchData* batch_data,
    std::vector<std::thread>& dl_threads,
    std::vector<std::thread>& nn_threads,
    std::vector<std::thread>& return_threads
);

game::IGame* get_game_instance(std::string game);

void queue_request(ThreadData* thread_data, Board& board, EvalRequest* request);