#include "./mc_tree.h"

MCTree::MCTree(){
    root = nullptr; //new MCNode(nullptr);
}

MCTree::~MCTree(){
    delete_tree(this->root);
}

void MCTree::delete_tree(MCNode * node){

    // base case
    if(node == nullptr){
        return;
    }

    // recursively delete children
    for(auto cp: node->children){
        delete_tree(cp.second);
    }

    // delete the node
    delete node;
}


void MCTree::move(game::move_id move_id){

    if(this->root == nullptr){
        return;
    }

    if(this->root->children[move_id] == nullptr ){
        // Delete root and all children
        delete_tree(this->root);

        // Create new node to be root
        this->root = nullptr;

    } else {
        // Get the new root
        MCNode *new_root = this->root->children[move_id];

        // Remove new root from root children
        this->root->children[move_id] = nullptr;

        // Remove parent from new root
        new_root->parent = nullptr;

        // Delete rest of tree
        delete_tree(this->root);

        // replace root
        this->root = new_root;
    }
}
