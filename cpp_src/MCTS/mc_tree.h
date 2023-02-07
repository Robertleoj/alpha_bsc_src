#pragma once
#include "./mc_node.h"

class MCTree {
public:
    MCNode * root;

    MCTree();
    ~MCTree();

    void move(game::move_id move_id);

private:
    void delete_tree(MCNode * node);
};
