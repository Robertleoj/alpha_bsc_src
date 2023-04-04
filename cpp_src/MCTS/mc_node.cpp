#include "./mc_node.h"
#include "../config/config.h"


/**
 * @brief Constructor for non-terminal nodes
 * 
 * @param parent Pointer to parent node
 * @param legal_moves List of legal moves from this node
 * @param move_from_parent  Move that led to this node from parent
 * @param nn_prior Prior probability distribution from neural network
 */
MCNode::MCNode(
    MCNode * parent, 
    std::vector<game::move_id> legal_moves,
    game::move_id move_from_parent
    // nn::move_dist * nn_prior
) : legal_moves(legal_moves),
    parent(parent),
    move_from_parent(move_from_parent)
{
    this->plays = 0;
    this->is_terminal = false;
    this->children = child_map();

    for(auto move_id : legal_moves){
        this->children[move_id] = nullptr;
    }

    // this->make_prior(nn_prior);
}

void MCNode::add_prior(nn::move_dist * nn_prior){
    this->make_prior(nn_prior);
}


/**
 * @brief Constructor for terminal nodes
 * 
 * @param parent Pointer to parent node
 * @param move_from_parent  Move that led to this node from parent
 */
MCNode::MCNode(
    MCNode * parent,
    game::move_id move_from_parent
) {
    this->parent = parent;
    this->plays = 0;
    this->is_terminal = true;
    this->move_from_parent = move_from_parent;
    this->children = child_map();
}

/**
 * @brief Add noise to the prior probability distribution of this node
 * 
 * @param noise The dirichlet noise to add
 */
void MCNode::add_noise(std::vector<double> noise){
    int i = 0;

    double lam = config::hp["dirichlet_lambda"].get<double>();

    for(auto &p : this->p_map){
        p.second = (1 - lam) * p.second + lam * noise[i];
    }
}

/**
 * @brief Create the prior from the prior given by the neural network - removes illegal moves
 * 
 * @param nn_prior The prior given by the neural network
 */
void MCNode::make_prior(nn::move_dist * nn_prior) {
    
    // in case we don't need to normalize
    if(nn_prior->size() == this->legal_moves.size()){
        this->p_map = nn::move_dist(*nn_prior);
        return;
    }

    this->p_map = nn::move_dist();

    double prob_sum = 0;

    for(auto mv_id: this->legal_moves){

        double prob = nn_prior->at(mv_id);
        this->p_map[mv_id] = prob;
        prob_sum += prob;
    }
    
    for(auto &kv: this->p_map){
        kv.second /= prob_sum;
    }
}

/**
 * @brief Visit counts for each move from this node
 * 
 * @return std::map<game::move_id, int> 
 */
std::map<game::move_id, int> MCNode::visit_count_map(){

    std::map<game::move_id, int> mp;

    for(auto &kv: this->children){
        int vc = 0;

        if(kv.second != nullptr){
            vc = kv.second->plays;
        }

        mp[kv.first] = vc;
    }

    return mp;
}

/**
 * @brief Update this node's value approximation - uses iterative mean update
 * 
 * @param v the value of the playout
 */
void MCNode::update_eval(double v){

    this->plays++;
    double n = (double) this->plays;
    this->value_approx = ((n - 1) / n) * this->value_approx + (1 / n) * v;

}
