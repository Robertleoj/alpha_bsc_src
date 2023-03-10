//
// Created by Yngvi Björnsson on 8.5.2022.
//

#pragma once
#include <cstdint>
#include "../games/game.h"

uint64_t perft(game::IGame& game, int depth);
void do_perft_test(game::IGame& game, int depth);
void do_flat_mc_test(game::IGame& game, int num_simulations);

