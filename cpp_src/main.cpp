#include <iostream>
#include <random>
#include <memory>
#include <string>
#include "base/types.h"
#include "games/connect4.h"
#include "games/breakthrough.h"
#include "misc/perft.h"
#include "MCTS/agent.h"
#include "NN/connect4_nn.h"
#include <torch/script.h>
// like import *
#include <torch/all.h>
#include <mariadb/conncpp.hpp>
#include "./DB/db.h"
#include "./self-play/selfplay.h"

// using namespace std;
using namespace game;
using namespace games;

using RunGameEntry = std::tuple<std::string,std::unique_ptr<game::IGame>,int, int>;


int main()
{
    // seed random
    srand(time(NULL));

    auto selfplayer = SelfPlay("connect4");
    selfplayer.self_play();


    return 0;
}
