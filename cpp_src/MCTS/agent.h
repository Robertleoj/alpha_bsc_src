#pragma once

#include "./mc_tree.h"
#include "../games/game.h"
#include "../base/types.h"
#include "../NN/nn.h"

class Agent {
public:
    Agent(game::IGame & game, pp::Player player, nn::NN* neural_net);

    ~Agent();
    
    nn::NN * neural_net;

    void update(int move_idx);
    void update_tree(int move_idx);
    game::move_iterator get_move(int playout_cap);
    void switch_sides();

private:
    game::IGame & game;
    pp::Player player;

    // Move ** move_buffer = nullptr;
    
    MCTree * tree = nullptr;
    // AgentState * state = nullptr;

    // Move * board_idx_to_move(board_idx b);

    // board_idx move_to_board_idx(Move m);

    std::pair<MCNode *, double> selection();

    void backpropagation(MCNode *node, double v);

    int get_current_best_move();

    double PUCT(MCNode *node, MCNode * childnode);
    
    double outcome_to_value(out::Outcome);

};

