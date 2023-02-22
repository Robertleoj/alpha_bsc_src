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
#include "./config/config.h"

// using namespace std;
using namespace game;
using namespace games;

using RunGameEntry = std::tuple<std::string,std::unique_ptr<game::IGame>,int, int>;


int main()
{
    std::cout << "PYTORCH VERSION "
              << TORCH_VERSION_MAJOR << '.' 
              << TORCH_VERSION_MINOR << '.' 
              << TORCH_VERSION_PATCH << std::endl;

    // std::ifstream f("../models/test.pt", std::ios::binary);
    // auto cu = std::make_shared<torch::CompilationUnit>();
 
    // torch::jit::import_ir_module(cu,"../models/test.pt", c10::Device("cuda"));
    // f.close();
    // seed random
    srand(time(NULL));
    // initialize config
    config::initialize();

    auto selfplayer = SelfPlay("connect4");
    selfplayer.self_play();


    return 0;
}
