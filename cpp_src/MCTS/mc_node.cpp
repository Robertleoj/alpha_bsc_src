#include "./mc_node.h"
#include <bits/stdc++.h>
#include "../config/config.h"


// non-terminal constructor
MCNode::MCNode(
    MCNode * parent, 
    std::vector<game::move_id> legal_moves,
    game::move_id move_from_parent,
    nn::NNOut nn_evaluation
){
    this->legal_moves = legal_moves;
    this->parent = parent;
    this->plays = 0;
    this->is_terminal = false;
    this->move_from_parent = move_from_parent;

    // this->value_approx = nn_evaluation.v;

    this->children = std::map<game::move_id, MCNode *>();
    for(auto move_id : legal_moves){
        this->children[move_id] = nullptr;
    }

    this->make_evaluation(&nn_evaluation);
}


MCNode::MCNode(
    MCNode * parent,
    game::move_id move_from_parent
) {
    this->parent = parent;
    this->plays = 0;
    this->is_terminal = true;
    this->move_from_parent = move_from_parent;
    this->children = std::map<game::move_id, MCNode *>();
}

void MCNode::add_noise(std::vector<double> noise){
    int i = 0;
    double lam = config::hp["dirichlet_lambda"].get<double>();
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
        int vc = 0;

        if(kv.second != nullptr){
            vc = kv.second->plays;
        }

        mp[kv.first] = vc;
    }

    return mp;
}

void MCNode::update_eval(double v){

    this->plays++;
    double n = (double) this->plays;
    this->value_approx = ((n - 1) / n) * this->value_approx + (1 / n) * v;

}
