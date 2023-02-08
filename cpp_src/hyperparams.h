#pragma once


namespace hp {
    const double PUCT_c = 4;
    const int batch_size = 512;
    const int num_parallel_games = 700;
    const int search_depth = 50;
    const int self_play_num_games = 10000;
    const double dirichlet_lambda = 0.25;
    const double dirichlet_alpha = 0.3;
}