#pragma once

#include "./mc_tree.h"
#include "../games/game.h"
#include "../base/types.h"
#include "../NN/nn.h"
#include <map>

typedef std::function<std::unique_ptr<nn::NNOut>(Board)> eval_f;

class Agent {
public:
    // variables
    MCTree * tree = nullptr;

    //constructor
    Agent(
        game::IGame * game, 
        eval_f eval_func,
        bool apply_noise = true,
        bool delete_on_move = true
    );

    // destructor
    ~Agent();

    // functions
    void update_tree(game::move_id move_id);
    void search(int playout_cap);
    std::map<game::move_id, int> root_visit_counts();
    std::pair<MCNode *, double> make_root();
    double outcome_to_value(out::Outcome);
    double eval_for_player(double eval, pp::Player player);
    double switch_eval(double eval);
    double PUCT(MCNode *node, game::move_id move);

private:
    //variables
    game::IGame * game;
    bool use_dirichlet_noise = true;
    bool delete_on_move;
    eval_f eval_func;

    //functions
    std::pair<MCNode *, double> selection();
    void backpropagation(MCNode *node, double v);
    game::move_id select_puct_move(MCNode * node);
    void apply_noise(MCNode * node);
    std::pair<MCNode *, double> make_node(MCNode * parent, game::move_id move_id);


};

