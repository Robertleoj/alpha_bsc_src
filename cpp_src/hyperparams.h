#pragma once



namespace hp {
    const double PUCT_c = 4;
    const int batch_size = 128;
    const int num_parallel_games = 256;
    const int search_depth = 500;
    const int self_play_num_games = 2000;
    const double dirichlet_lambda = 0.25;
    const double dirichlet_alpha = 0.3;
}
