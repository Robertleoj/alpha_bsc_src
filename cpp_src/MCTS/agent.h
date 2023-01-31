#pragma once

#include "./mc_tree.h"
#include "../games/game.h"
// #include "../games/move.h"
#include "../base/types.h"

class Agent {
public:
    Agent(game::IGame & game, pp::Player player);

    ~Agent();

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

    MCNode * selection();

    out::Outcome simulation(MCNode * node);

    void backpropagation(MCNode *node, out::Outcome sim_res);

    int get_current_best_move();

    double UCT(MCNode *node, MCNode * childnode);

};

