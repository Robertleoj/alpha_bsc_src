#pragma once

#include "./mc_tree.h"
#include "../games/game.h"
#include "../base/types.h"
#include "../NN/nn.h"
#include <map>

typedef std::function<std::unique_ptr<nn::NNOut>(Board)> eval_f;

class Agent {
public:
    MCTree * tree = nullptr;
    eval_f eval_func;
    bool use_dirichlet_noise = true;


    Agent(
        game::IGame * game, 
        pp::Player player, 
        eval_f eval_func,
        bool apply_noise = true
    );

    ~Agent();
    
    void update(int move_idx);

    void update_tree(game::move_id move_id);

    void search(int playout_cap);
    std::map<game::move_id, int> root_visit_counts();

    void switch_sides();
    double outcome_to_value(out::Outcome);
    double eval_for_player(double eval, pp::Player player);
    double switch_eval(double eval);

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

