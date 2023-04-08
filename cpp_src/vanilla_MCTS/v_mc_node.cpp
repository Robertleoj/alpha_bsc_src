#include "./v_mc_node.h"
#include <bits/stdc++.h>


VMCNode::VMCNode(
    VMCNode * parent, 
    game::move_id move_from_parent,
    std::vector<game::move_id> legal_moves
){
    this->parent = parent;
    this->plays = 0;
    this->wins = 0;
    this->move_from_parent = move_from_parent;
    this->is_terminal = false;
    for(auto move : legal_moves){
        this->children[move] = nullptr;
    }
}



