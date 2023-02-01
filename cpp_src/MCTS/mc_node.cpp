#include "./mc_node.h"
#include <bits/stdc++.h>


// non-terminal constructor
MCNode::MCNode(
    MCNode * parent, 
    game::MovelistPtr move_list, 
    int idx_in_parent,
    nn::NNOut nn_evaluation
){
    this->parent = parent;
    this->plays = 1;
    this->is_terminal = false;
    this->idx_in_parent = idx_in_parent;
    this->move_list = std::move(move_list);
    this->value_approx = nn_evaluation.v;

    this->children = std::vector<MCNode *>(this->move_list->get_size(), nullptr);

    this->make_evaluation(&nn_evaluation);
}

MCNode::MCNode(
    MCNode * parent,
    int idx_in_parent,
    double terminal_eval
) {
    this->parent = parent;
    this->plays = 1;
    this->is_terminal = true;
    this->idx_in_parent = idx_in_parent;
    this->value_approx = terminal_eval;
}

void MCNode::make_evaluation(nn::NNOut * nn_evaluation) {
    
    this->p = std::vector<double>(this->children.size());

    auto &mp = nn_evaluation->p;
    
    double prob_sum = 0;

    for(int i = 0; i < this->children.size(); i++){
        auto mv = this->move_list->begin() + i;
        auto mv_id = this->move_list->as_move_id(mv);
        double prob = mp.at(mv_id);
        this->p[i] = prob;
        prob_sum += prob;
    }
    
    for(int i = 0; i < this->p.size(); i++){
        this->p[i] /= prob_sum;
    }
}

void MCNode::update_eval(double v){

    this->plays++;
    int n = this->plays;
    this->value_approx = ((n - 1) / n) * this->value_approx + (1 / n) * v;

}
