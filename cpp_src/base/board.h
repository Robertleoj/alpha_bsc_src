//
// Created by Yngvi Björnsson on 30.1.2023.
//

#pragma once
#include <vector>
#include "types.h"
#include "bitboard.h"

struct Board {
    Player to_move;
    std::vector<bb::Bitboard> bbs;
};
