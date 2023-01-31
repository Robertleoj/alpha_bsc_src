#pragma once
// #include <move.h>
#include <map>
#include "../games/game.h"
#include "../games/move.h"
#include "../NN/nn.h"
#include <vector>

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
    // std::map<int, MCNode*> children;

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
    
    void update_eval(double v);

private:
    void make_evaluation(nn::NNOut * nn_evaluation);

};