#pragma once

#include "./mc_tree.h"
#include "../games/game.h"
#include "../base/types.h"
#include "../NN/nn.h"
#include <map>

class Agent {
public:
    nn::NN * neural_net;
    MCTree * tree = nullptr;


    Agent(game::IGame * game, pp::Player player, nn::NN* neural_net);

    ~Agent();
    
    void update(int move_idx);

    void update_tree(game::move_id move_id);

    void search(int playout_cap);
    std::map<game::move_id, int> root_visit_counts();

    void switch_sides();
    double outcome_to_value(out::Outcome);

private:
    game::IGame * game;
    pp::Player player;

    // Move ** move_buffer = nullptr;
    
    // AgentState * state = nullptr;

    // Move * board_idx_to_move(board_idx b);

    // board_idx move_to_board_idx(Move m);

    std::pair<MCNode *, double> selection();

    void backpropagation(MCNode *node, double v);

    int get_current_best_move();

    double PUCT(MCNode *node, MCNode * childnode);
    

};

