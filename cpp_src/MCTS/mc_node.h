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

    MCNode * parent;

    bool is_terminal;
    std::vector<game::move_id> legal_moves;
    game::move_id move_from_parent;
    std::map<game::move_id, double> p_map;
    std::map<game::move_id, MCNode*> children;

    MCNode(
        MCNode * parent, 
        std::vector<game::move_id> legal_moves,
        game::move_id move_from_parent, 
        nn::NNOut nn_evaluation
    );
    
    // terminal constructor
    MCNode(
        MCNode * parent,
        game::move_id move_from_parent,
        double terminal_eval
    );

    void update_eval(double v);
    std::map<game::move_id, int> visit_count_map();

    void add_noise(std::vector<double> dirich_noise);

private:
    void make_evaluation(nn::NNOut * nn_evaluation);

};