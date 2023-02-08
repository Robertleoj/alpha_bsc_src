#include "./mc_node.h"
#include <bits/stdc++.h>
#include "../hyperparams.h"


// non-terminal constructor
MCNode::MCNode(
    MCNode * parent, 
    std::vector<game::move_id> legal_moves,
    game::move_id move_from_parent,
    nn::NNOut nn_evaluation
){
    this->legal_moves = legal_moves;
    this->parent = parent;
    this->plays = 1;
    this->is_terminal = false;
    this->move_from_parent = move_from_parent;

    this->value_approx = nn_evaluation.v;

    this->children = std::map<game::move_id, MCNode *>();
    for(auto move_id : legal_moves){
        this->children[move_id] = nullptr;
    }

    this->make_evaluation(&nn_evaluation);
}


MCNode::MCNode(
    MCNode * parent,
    game::move_id move_from_parent,
    double terminal_eval
) {
    this->parent = parent;
    this->plays = 1;
    this->is_terminal = true;
    this->move_from_parent = move_from_parent;
    this->value_approx = terminal_eval;
    this->children = std::map<game::move_id, MCNode *>();
}

void MCNode::add_noise(std::vector<double> noise){
    int i = 0;
    double lam = hp::dirichlet_lambda;
    for(auto &p : this->p_map){
        p.second = (1 - lam) * p.second + lam * noise[i];
    }
}

void MCNode::make_evaluation(nn::NNOut * nn_evaluation) {
    
    this->p_map = std::map<game::move_id, double>();

    auto mp = nn_evaluation->p;
    
    double prob_sum = 0;

    for(auto mv_id: this->legal_moves){

        double prob = mp.at(mv_id);
        this->p_map[mv_id] = prob;
        prob_sum += prob;
    }
    
    for(auto &kv: this->p_map){
        kv.second /= prob_sum;
    }
}

std::map<game::move_id, int> MCNode::visit_count_map(){

    std::map<game::move_id, int> mp;

    for(auto &kv: this->children){
        mp[kv.first] = kv.second->plays;
    }

    return mp;
}

void MCNode::update_eval(double v){

    this->plays++;
    int n = this->plays;
    this->value_approx = ((n - 1) / n) * this->value_approx + (1 / n) * v;

}
