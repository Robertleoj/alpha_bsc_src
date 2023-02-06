#pragma once
#include <vector>
#include "types.h"
#include "bitboard.h"

struct Board {
    Player to_move;
    std::vector<bb::Bitboard> bbs;
};
