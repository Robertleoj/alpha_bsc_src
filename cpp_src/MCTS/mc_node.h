#pragma once
// #include <move.h>
#include <map>
#include "../games/game.h"
#include "../games/move.h"
#include <vector>

class MCNode{
public:
    int plays;
    double wins;
    MCNode * parent;
    bool is_terminal;

    int idx_in_parent;

    game::MovelistPtr move_list;

    std::vector<MCNode*> children;
    // std::map<int, MCNode*> children;

    ~MCNode();

    MCNode(MCNode * parent, game::MovelistPtr, int);
};