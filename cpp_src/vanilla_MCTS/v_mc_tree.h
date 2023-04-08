#pragma once
#include "./v_mc_node.h"
#include "../games/game.h"

class VMCTree {
public:
    VMCNode * root;

    VMCTree();
    ~VMCTree();

    void move(game::move_id);

private:
    void delete_tree(VMCNode * node);
};