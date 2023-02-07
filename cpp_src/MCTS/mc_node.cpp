#include "./mc_node.h"
#include <bits/stdc++.h>


MCNode::~MCNode(){
    this->move_list.release();

    // deallocation does not work - put this when it does
    // this->move_list.reset();
}

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

int MCNode::move_idx_of(game::move_id move_id){
    int move_idx = 0;

    for(
        auto a = this->move_list->begin(); 
        a != this->move_list->end();
        a++
    ) {
        if(this->move_list->as_move_id(a) == move_id){
            return move_idx;
        }
        move_idx++;
    }
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
    this->children = std::vector<MCNode *>();
}

void MCNode::make_evaluation(nn::NNOut * nn_evaluation) {
    
    this->p = std::vector<double>(this->children.size());

    auto mp = nn_evaluation->p;
    
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

std::map<game::move_id, int> MCNode::visit_count_map(){
    std::map<game::move_id, int> mp;
    for(int i = 0; i < this->children.size(); i++){
        auto mv = this->move_list->begin() + i;
        mp[this->move_list->as_move_id(mv)] = this->children[i]->plays;
    }
    return mp;
}

void MCNode::update_eval(double v){

    this->plays++;
    int n = this->plays;
    this->value_approx = ((n - 1) / n) * this->value_approx + (1 / n) * v;

}
