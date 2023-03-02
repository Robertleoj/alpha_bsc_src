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
    typedef std::map<game::move_id, MCNode*> child_map;

    // constructors
    //// non-terminal constructor
    MCNode(
        MCNode * parent, 
        std::vector<game::move_id> legal_moves,
        game::move_id move_from_parent, 
        nn::move_dist * nn_prior
    );
    
    //// terminal constructor
    MCNode(
        MCNode * parent,
        game::move_id move_from_parent
    );

    // variables
    int plays;
    double value_approx;
    MCNode * parent;
    bool is_terminal;
    std::vector<game::move_id> legal_moves;
    game::move_id move_from_parent;
    child_map children;
    nn::move_dist p_map;


    // methods
    void update_eval(double v);

    std::map<game::move_id, int> visit_count_map();

    void add_noise(std::vector<double> dirich_noise);

private:
    // methods
    void make_prior(nn::move_dist * nn_prior);

};
