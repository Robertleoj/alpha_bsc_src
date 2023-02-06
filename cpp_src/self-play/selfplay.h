#pragma once

#include "../MCTS/agent.h"
#include "../DB/db.h"
#include "../NN/nn.h"
#include <string>
#include "../base/types.h"
#include <queue>
#include <mutex>

typedef std::pair<int, Board> eval_request;

class SelfPlay {
public:
    std::unique_ptr<nn::NN> neural_net;
    std::unique_ptr<db::DB> db;
    std::string game;

    SelfPlay(std::string game);

    void play_game();

    void self_play();
    void thread_play(
        int thread_idx, 
        std::queue<eval_request> * eval_q,
        std::mutex * q_mutex
    );
};