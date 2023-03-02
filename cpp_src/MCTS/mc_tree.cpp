#include "./mc_tree.h"

MCTree::MCTree(){
    root = nullptr; 
}

MCTree::~MCTree(){
    delete_tree(this->root);
}

/**
 * @brief Recursively delete the tree starting at node
 * 
 * @param node 
 */
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


/**
 * @brief Move the root of the tree to the child node with move_id and delete the rest of the tree
 * 
 * @param move_id 
 */
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
