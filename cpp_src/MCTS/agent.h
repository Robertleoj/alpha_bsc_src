#pragma once

#include "./mc_tree.h"
#include "../games/game.h"
#include "../base/types.h"
#include "../NN/nn.h"
#include <map>

typedef std::function<std::unique_ptr<nn::NNOut>(
    Board, std::vector<game::move_id> *
)> eval_f;

class Agent {
public:
    // variables
    MCTree * tree = nullptr;
    int num_playouts = 0;
    int max_playouts = 0;

    //constructor
    Agent(
        game::IGame * game, 
        bool apply_noise = true,
        bool delete_on_move = true
    );

    // destructor
    ~Agent();

    // functions
    void update_tree(game::move_id move_id);
    void search(int playout_cap, eval_f eval_func);
    std::map<game::move_id, int> root_visit_counts();
    double outcome_to_value(out::Outcome);
    double eval_for_player(double eval, pp::Player player);
    double switch_eval(double eval);
    double PUCT(MCNode *node, game::move_id move);
    std::pair<bool, Board> step(std::unique_ptr<nn::NNOut>);
    std::pair<bool, Board> init_mcts(int max_playouts);
    std::vector<game::move_id> * node_legal_moves();

private:
    //variables
    game::IGame * game;
    bool use_dirichlet_noise = true;
    bool delete_on_move;
    MCNode * node_to_eval = nullptr;
    double puct_c;

    //functions
    std::tuple<MCNode *, double, bool> selection();
    void backpropagation(MCNode *node, double v);
    game::move_id select_puct_move(MCNode * node);
    void apply_noise(MCNode * node);
    std::tuple<MCNode *, double, bool> make_node(MCNode * parent, game::move_id move_id);
    std::tuple<MCNode *, double, bool> make_root();



};

