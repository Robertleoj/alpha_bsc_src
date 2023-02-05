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

// using namespace std;
using namespace game;
using namespace games;

using RunGameEntry = std::tuple<std::string,std::unique_ptr<game::IGame>,int, int>;

void play_a_game(game::IGame& game)
{
    game.display(std::cout);
    std::cout << std::endl;
    auto connect4_nn = nn::Connect4NN("../models/test_model.pt");
    

    auto agent = Agent(game, pp::First, (nn::NN*)&connect4_nn);
    
    bool agent_turn = true;
    int num_moves = 0;

    while(!game.is_terminal()) {

        auto mv = agent.get_move(100);
        std::cout << "Move :" << game.move_as_str(mv) << std::endl;

        // } else {
        //     auto ml = game.moves();
        //     int rand_idx = rand() % ml->get_size();
        //     auto mv = ml->begin() + rand_idx;
        //     game.make(mv);
        //     agent.update_tree(rand_idx);
        //     cout << "Move :" << game.move_as_str(mv) << endl;
        // }
        
        game.display(std::cout);
        
        agent_turn = !agent_turn;

        std::cout << std::endl;
        std::cout << "Move " << ++num_moves << std::endl;

        std::cout << std::endl;
    }
    std::cout << str(game.outcome(First)) << std::endl;
}



int main()
{
    // test putting a tensor on gpu
    // auto t = torch::ones({1, 2, 3}).cuda();

    srand(time(NULL));

    auto conn = db::DB();

    auto res = conn.conn->prepareStatement("select * from generations")->executeQuery();

    while(res->next()){
        std::cout << "Generation num: " << res->getInt("generation_num") << std::endl;
    }


    auto game = std::make_unique<Connect4>();
    // auto game = std::make_unique<Breakthrough>();
    play_a_game(*game.get());

    auto t = torch::randn({10, 10});

    // std::cout << "t1" << std::endl;
    // std::cout << t << std::endl;

    std::stringstream ss;
    
    torch::save(t,ss);

    at::Tensor t2;

    torch::load(t2, ss);

    std::cout << "t2" << std::endl;
    std::cout << t2 << std::endl;

    return 0;
}
