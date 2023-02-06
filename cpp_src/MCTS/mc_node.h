#pragma once
// #include <move.h>
#include <map>
#include "../games/game.h"
#include "../games/move.h"
#include "../NN/nn.h"
#include <vector>
#include <map>

class MCNode{
public:
    int plays;
    double value_approx;
    
    std::vector<double> p;

    MCNode * parent;
    bool is_terminal;

    int idx_in_parent;

    game::MovelistPtr move_list;

    std::vector<MCNode*> children;

    MCNode(
        MCNode * parent, 
        game::MovelistPtr, 
        int idx_in_parent, 
        nn::NNOut nn_evaluation
    );
    
    // terminal constructor
    MCNode(
        MCNode * parent,
        int idx_in_parent,
        double terminal_eval
    );

    ~MCNode();

    int move_idx_of(game::move_id);
    
    void update_eval(double v);
    std::map<game::move_id, int> visit_count_map();

private:
    void make_evaluation(nn::NNOut * nn_evaluation);

};