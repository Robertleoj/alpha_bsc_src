#include "./mc_node.h"
#include <bits/stdc++.h>


MCNode::MCNode(MCNode * parent, game::MovelistPtr move_list, int idx_in_parent){
    this->parent = parent;
    this->plays = 0;
    this->wins = 0;
    is_terminal = false;
    this->move_list = std::move(move_list);
    this->idx_in_parent = idx_in_parent;
    this->children = std::vector<MCNode *>(this->move_list->get_size(), nullptr);
}


MCNode::~MCNode(){
    // Nothing to destroy
}
