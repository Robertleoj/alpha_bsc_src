#include "./v_mc_tree.h"

VMCTree::VMCTree(){
    this->root = nullptr; //new MCNode(nullptr);
}

VMCTree::~VMCTree(){
    delete_tree(this->root);
}

void VMCTree::delete_tree(VMCNode * node){
    // base case
    if(node == nullptr){
        return;
    }

    // recursively delete children
    for(auto &c: node->children){
        delete_tree(c.second);
    }

    // delete the node
    delete node;
}


void VMCTree::move(game::move_id move_id){

    // delete_tree(this->root);
    // this->root = nullptr;
    // return;

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
        VMCNode *new_root = this->root->children[move_id];

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