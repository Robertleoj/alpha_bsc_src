#pragma once

#include <string>
#include <vector>
#include "../NN/connect4_nn.h"
#include "../NN/breakthrough_nn.h"
#include "../games/move.h"
#include "../games/game.h"
#include "../MCTS/agent.h"
#include "../games/breakthrough.h"
#include "../games/connect4.h"
#include "../NN/nn.h"


class Competitor {
public:
    Competitor(
        std::string run_name, 
        std::string game_name, 
        int generation, 
        int num_agents,
        int num_playouts
    );

    void update(std::vector<std::string> moves);

    std::vector<std::string> make_and_get_moves();
    std::vector<double> get_results();

private:
    std::vector<game::IGame *> games;
    std::vector<Agent *> agents;
    nn::NN * neural_net;
    int num_playouts;
    int num_dead;

    std::vector<double> results;

    std::vector<bool> dead;
};