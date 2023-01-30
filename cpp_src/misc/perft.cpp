//
// Created by Yngvi Björnsson on 8.5.2022.
//
#include <iostream>
#include <random>
#include "perft.h"


uint64_t perft(game::IGame& game, int depth)
{
    if (depth == 0 || game.is_terminal()) {
        return 1ULL;
    }
    uint64_t nodes = 0;
    auto ml = game.moves();
    for (auto it = ml->begin(); it != ml->end(); ++it) {
        game.make(it);
        nodes += perft(game, depth - 1);
        game.retract(it);
    }
    return nodes;
}


uint64_t do_a_simulation(game::IGame& game, game::move_iterator it, std::default_random_engine& generator)
{
    uint64_t nodes = 1;
    game.push();
    game.make(it);
    while(!game.is_terminal()) {
        auto ml = game.moves();
        std::uniform_int_distribution<int> distribution(0, ml->get_size()-1);
        int n = distribution(generator);
        game.make(ml->begin() + n);
        ++nodes;
    }
    game.pop();
    return nodes;
}

uint64_t flat_mc(game::IGame& game, int num_simulations)
{
    struct Node {
        game::move_iterator it;
        uint64_t n;
        double sum;
    };

    std::vector<Node> children;

    auto ml = game.moves();

    children.reserve(ml->get_size());

    for (auto it=ml->begin(); it != ml->end(); ++it) {
        children.push_back({it, 0, 0.0});
    }

    std::default_random_engine generator(42);
    
    uint64_t nodes = 0ULL;

    for (int i = 0; i < num_simulations; ++i) {
        std::uniform_int_distribution<int> distribution(0, children.size()-1);
        int n = distribution(generator);
        nodes += do_a_simulation(game, children[n].it, generator);
    }

    return nodes;
}


[[maybe_unused]] void do_perft_test(game::IGame& game, int depth)
{
    utils::Timer timer;
    timer.start();
    auto nodes = perft(game, depth);
    timer.stop();
    auto ms = timer.duration_ms();
    std::cout << nodes << " nodes  " << ms << " ms.";
    if (ms > 0) {
        auto nps = (uint64_t) ((double)nodes / ((double)ms/1000.0));
        std::cout << "  " << nps << " nps.";
    }
    std::cout << std::endl;
}


[[maybe_unused]] void do_flat_mc_test(game::IGame& game, int num_simulations)
{
    utils::Timer timer;
    timer.start();
    auto nodes = flat_mc(game, num_simulations);
    timer.stop();
    auto ms = timer.duration_ms();
    std::cout << num_simulations <<  " simulations  " << nodes << " nodes  " << ms << " ms.";
    if (ms > 0) {
        auto sps = (uint64_t) ((double)num_simulations / ((double)ms/1000.0));
        auto nps = (uint64_t) ((double)nodes / ((double)ms/1000.0));
        std::cout << "  " << sps << " sps." << "  " << nps << " nps.";
    }
    std::cout << std::endl;
}