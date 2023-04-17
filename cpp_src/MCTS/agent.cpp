#include "./agent.h"
#include <math.h>
#include <random>
#include <stdexcept>
#include "../utils/utils.h"
#include "../config/config.h"


Agent::Agent(
    game::IGame * game, 
    bool apply_noise,
    bool delete_on_move
) : game(game), 
    use_dirichlet_noise(apply_noise), 
    delete_on_move(delete_on_move) 
{  
    this->tree = new MCTree();
    this->puct_c = config::hp["PUCT_c"].get<double>();
}

Agent::~Agent(){
    if (this->tree != nullptr) {
        delete this->tree;
    }
}


/**
 * @brief Update the tree. Depending on the config, this can be done by deleting the tree and creating a new one, or by moving the root node to the new state.
 * 
 * @param move_id 
 */
void Agent::update_tree(
    game::move_id move_id
){

    if(this->delete_on_move){
        delete this->tree;
        this->tree = new MCTree();
    } else {

        this->tree->move(move_id);

        if(!this->game->is_terminal() && this->tree->root != nullptr){
            this->apply_noise(this->tree->root);
        }
    }
}


/**
 * @brief PUCT(s, a) = V + c * P(a|s)(sqrt{N} / (1 + n))
 *  V = evaluations for the child node
 *  n = number of simulations for child node
 *  N = simulations for current node
 *  c = hp
 * @param node
 * @param move
 * @returns PUCT value
*/
double Agent::PUCT(MCNode * node, game::move_id move){

    double P = node->p_map[move];
    double c = this->puct_c;
    double N = node->plays;

    double V = 0;
    double n = 0;

    MCNode * childnode = node->children[move];

    if(childnode != nullptr){
        /*
         value-approx is the evaluation of the node
         with respect to the player whose move it is in that node
         thus we reverse the evaluation
        */
        V = this->switch_eval(childnode->value_approx);
        n = childnode->plays;
    }

    return V + c *P * sqrt(N) / (1 + n);
}

double Agent::outcome_to_value(out::Outcome oc){
    switch(oc){
        case out::Outcome::Loss:
            return -1;
        case out::Outcome::Win:
            return 1;
        case out::Outcome::Tie:
            return 0;
        default:
            throw std::runtime_error("Undecided outcome");
    }
}

/**
 * @brief Evaluation from the perspective of the player, assumes v is from the perspective of the first player
 * 
 * @param v 
 * @param player 
 * @return double 
 */
double Agent::eval_for_player(double v, pp::Player player){
    if(player == pp::First){
        return v;
    } else {
        return -v;
    }
}

double Agent::switch_eval(double v) {
    return -v;
}


/**
 * @brief Create new node - assumes the game is in the state the node should represent
 * 
 * @param parent 
 * @param move_id 
 * @return std::pair<MCNode *, double> 
 */
std::tuple<MCNode *, double, bool> Agent::make_node(MCNode * parent, game::move_id move_id){
    MCNode * new_node = nullptr;
    double v;
    bool terminal;


    if(game->is_terminal()){
        v = this->outcome_to_value(game->outcome());
        terminal = true;

        new_node = new MCNode(
            parent,
            move_id
        );

    } else {

        v = 0;
        terminal = false;

        // Make new node 
        new_node = new MCNode(
            parent,
            this->game->moves(),
            move_id
        );
    }

    if(parent != nullptr){
        parent->children[move_id] = new_node;
    }

    return std::make_tuple(new_node, v, terminal);
}


/**
 * @brief Applies dirichlet noise to the root node
 * 
 * @param node 
 */
void Agent::apply_noise(MCNode * node){

    if(!this->use_dirichlet_noise){
        return;
    }

    auto noise = utils::dirichlet_dist(
        config::hp["dirichlet_alpha"].get<double>(), 
        node->legal_moves.size()
    );

    node->add_noise(noise);
}


/**
 * @brief Returns the move id of the move to be selected/explored
 * 
 * @param node 
 * @return selected move id
 */
game::move_id Agent::select_puct_move(MCNode * node){
    double max_uct = -100000;

    auto &legal_moves = node->legal_moves;

    game::move_id best_move;

    for(auto mv: legal_moves){
        double uct = PUCT(node, mv);

        if(uct > max_uct){
            max_uct = uct;
            best_move = mv;
        }
    }

    return best_move;
}


/**
 * @brief Create root node - applies noise if applicable
 * 
 * @return pair with the new root and its evaluation
 */
std::tuple<MCNode *, double, bool> Agent::make_root(){
    auto ret = this->make_node(nullptr, -1);

    auto [root, value, _] = ret;

    this->tree->root = root;

    // if(!this->game->is_terminal()){
    //     this->apply_noise(root);
    // }

    return ret;
}

/**
 * @brief Selection phase of MCTS
 * 
 * @return Pair with the selected node and its evaluation from the perspective of the player whose move it is
 */
std::tuple<MCNode *, double, bool> Agent::selection(){

    // If root doesn't exist, create it
    if (this->tree->root == nullptr){
        return this->make_root();
    }

    MCNode * current_node = tree->root;

    while(true){

        // if the current node is terminal, the selection phase is over, and there is no expansion.
        if(current_node->is_terminal){
            return std::make_tuple(current_node, current_node->value_approx, true);
        }

        // find move with highest puct
        game::move_id best_move = this->select_puct_move(current_node);

        MCNode * next_node = current_node->children[best_move];

        game->make(best_move);

        if(next_node == nullptr){
            // if the next node is null, we have reached a leaf node, and we need to expand it
            return this->make_node(current_node, best_move);

        } else {
            //continue selection
            current_node = next_node;
        }
    }
}



/**
 * @brief Backpropagation phase of MCTS
 * 
 * @param node The node to start backpropagation from
 * @param v The evaluation of the node
 */
void Agent::backpropagation(MCNode * node, double v){

    while(true){

        node->update_eval(v);

        if (node->parent == nullptr) {
            break;
        } else {
            this->game->retract(node->move_from_parent);
            node = node->parent;
            v = this->switch_eval(v);
        }

    }
}

/**
 * @brief Performs a search with the given playout cap
 * 
 * @param playout_cap 
 * @param eval_func
 */
void Agent::search(int playout_cap, eval_f eval_func){

    auto [done, board] = this->init_mcts(playout_cap);
    while(!done){
        auto evaluation = eval_func(
            board, 
            &this->node_to_eval->legal_moves
        );
        std::tie(done, board) = this->step(std::move(evaluation));
    }
}

std::pair<bool, Board> Agent::step(std::unique_ptr<nn::NNOut> evaluation){

    // get node to evaluate
    if(this->node_to_eval == nullptr){
        throw std::runtime_error("No node to evaluate, did you forget to call init_mcts()?");
    }

    // update node
    this->node_to_eval->add_prior(&evaluation->p);

    if(this->node_to_eval == this->tree->root){
        if (this->use_dirichlet_noise){
            this->apply_noise(this->node_to_eval);
        }
    }


    // get new node to update
    bool terminal;
    MCNode * new_node = this->node_to_eval;
    double v = evaluation->v;
    do {
        // backprop
        this->backpropagation(new_node, v);
        this->num_playouts++;
        if(this->num_playouts >= this->max_playouts){
            return std::make_pair(true, this->game->get_board());
        }
        std::tie(new_node, v, terminal) = this->selection();

    } while(terminal);

    this->node_to_eval = new_node;

    // return board of node
    return std::make_pair(false,this->game->get_board());
}

std::vector<game::move_id> * Agent::node_legal_moves(){
    return &this->node_to_eval->legal_moves;
}

std::pair<bool, Board> Agent::init_mcts(int max_playouts){
    this->max_playouts = max_playouts;
    this->num_playouts = 0;

    auto [new_node, v, terminal] = this->selection();

    while(terminal) {
        // backprop
        this->backpropagation(new_node, v);
        this->num_playouts++;
        if(this->num_playouts >= this->max_playouts){
            return std::make_pair(true, this->game->get_board());
        }
        auto res = this->selection();
        std::tie(new_node, v, terminal) = res;
    }

    this->node_to_eval = new_node;

    // return board of node
    return std::make_pair(false, this->game->get_board());
}

/**
 * @brief Returns the visit counts of the root node
 * 
 * @return std::map<game::move_id, int> 
 */
std::map<game::move_id, int> Agent::root_visit_counts(){
    return this->tree->root->visit_count_map();
}

game::move_id Agent::best_move() {

    auto visit_counts = this->root_visit_counts();

    game::move_id best_move;
    int best_visit_count = -1;

    for(auto &p : visit_counts) {
        if(p.second > best_visit_count){
            best_move = p.first;
            best_visit_count = p.second;
        }
    }

    return best_move;
}
