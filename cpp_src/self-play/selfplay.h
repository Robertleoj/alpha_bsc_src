#pragma once

#include "../MCTS/agent.h"
#include "../DB/db.h"
#include "../NN/nn.h"
#include <string>


class SelfPlay {
public:
    std::unique_ptr<nn::NN> neural_net;
    std::unique_ptr<db::DB> db;
    std::string game;

    SelfPlay(std::string game);

    void play_game();
};