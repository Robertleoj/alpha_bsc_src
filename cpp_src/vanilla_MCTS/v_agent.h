#pragma once

#include "./v_mc_tree.h"
#include "../games/game.h"
// #include "../games/move.h"
#include "../base/types.h"

class VAgent {
public:
    VAgent(game::IGame * game);

    ~VAgent();

    void update_tree(game::move_id);
    game::move_id get_move(int playout_cap);

private:
    game::IGame * game;
    
    VMCTree * tree = nullptr;

    VMCNode * selection();

    out::Outcome simulation(VMCNode * node);

    void backpropagation(VMCNode *node, out::Outcome sim_res);

    game::move_id get_current_best_move();

    double UCT(VMCNode *node, VMCNode * childnode);

};