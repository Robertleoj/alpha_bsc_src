#pragma once
// #include <move.h>
#include <map>
#include "../games/game.h"
#include "../games/move.h"
#include <vector>

class VMCNode{
public:
    int plays;
    double wins;
    game::move_id move_from_parent;

    VMCNode * parent;

    bool is_terminal;

    std::map<game::move_id, VMCNode*> children;

    VMCNode(
        VMCNode * parent, 
        game::move_id move_from_parent,
        std::vector<game::move_id>
    );
};