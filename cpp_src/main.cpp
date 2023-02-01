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

// using namespace std;
using namespace game;
using namespace games;

using RunGameEntry = std::tuple<std::string,std::unique_ptr<game::IGame>,int, int>;

void play_a_game(game::IGame& game)
{
    game.display(std::cout);
    std::cout << std::endl;
    auto connect4_nn = nn::Connect4NN();
    
    

    auto agent = Agent(game, pp::First, (nn::NN*)&connect4_nn);
    
    bool agent_turn = true;
    int num_moves = 0;

    while(!game.is_terminal()) {

        auto mv = agent.get_move(10000);
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
    auto t = torch::ones({1, 2, 3});

    srand(time(NULL));

    auto game = std::make_unique<Connect4>();
    // auto game = std::make_unique<Breakthrough>();
    play_a_game(*game.get());

    return 0;
}
