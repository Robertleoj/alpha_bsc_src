//
// Created by Yngvi Björnsson on 30.1.2023.
//

#ifndef BSCPROJECTABG_BOARD_H
#define BSCPROJECTABG_BOARD_H
#include <vector>
#include "types.h"
#include "bitboard.h"

struct Board {
    Player to_move;
    std::vector<bb::Bitboard> bbs;
};
#endif //BSCPROJECTABG_BOARD_H
