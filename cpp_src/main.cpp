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
    // test putting a tensor on gpu
    // auto t = torch::ones({1, 2, 3}).cuda();

    // seed random
    srand(time(NULL));

    auto selfplayer = SelfPlay("connect4");
    selfplayer.self_play();


    // auto t = torch::randn({10, 10});

    // std::cout << "t1" << std::endl;
    // std::cout << t << std::endl;

    // std::stringstream ss;
    
    // torch::save(t,ss);

    // at::Tensor t2;

    // torch::load(t2, ss);

    // std::cout << "t2" << std::endl;
    // std::cout << t2 << std::endl;

    return 0;
}
