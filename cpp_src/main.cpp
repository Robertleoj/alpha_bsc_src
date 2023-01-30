#include <iostream>
#include <random>
#include <memory>
#include <string>
#include "base/types.h"
#include "games/connect4.h"
#include "games/breakthrough.h"
#include "misc/perft.h"

using namespace std;
using namespace game;
using namespace games;

using RunGameEntry = tuple<string,unique_ptr<game::IGame>,int, int>;


[[maybe_unused]]
void play_a_game(game::IGame& game)
{
    std::default_random_engine generator(42);
    game.display(cout);
    cout << endl;
    while(!game.is_terminal()) {
        auto ml = game.moves();
        for (auto it = ml->begin(), end = ml->end(); it != end; ++it) {
            cout << ' ' << game.move_as_str(it);
        }
        cout << endl;
        std::uniform_int_distribution<int> distribution(0, ml->get_size()-1);
        auto n = distribution(generator);
        game.make(ml->begin() + n);
        game.display(cout);
        cout << endl;
    }
    cout << str(game.outcome(First)) << endl;
}


[[maybe_unused]]
int play_moves(game::IGame& game, std::vector<std::string>& moves)
{
    int n = 0;
    for (const auto& move_str : moves) {
        if (game::make_move_if_legal(game, move_str)) {
            ++n;
        }
        else { break; }
    }
    return n;
}


[[maybe_unused]]
void test_performance(const RunGameEntry& g)
{
    do_perft_test( *get<1>(g), get<2>(g));
    do_flat_mc_test( *get<1>(g), get<3>(g));
}


[[maybe_unused]]
vector<string> split(const string &s, char delim) {
    vector<string> result;
    stringstream ss (s);
    string item;
    while (getline (ss, item, delim)) {
        result.push_back (item);
    }
    return result;
}


[[maybe_unused]]
int main()
{
    std::cout << "BScProject Abstract Board Games (8x8)" << std::endl;

    const int SEARCH_DEPTH_C4 = 10;
    const int SEARCH_DEPTH_BT = 6;
    const int NUM_SIMULATIONS = 100000;

    std::vector<RunGameEntry> run_games;

    run_games.emplace_back(
        "Connect4",      
        make_unique<Connect4>(), 
        SEARCH_DEPTH_C4, 
        NUM_SIMULATIONS
    );

    run_games.emplace_back(
        "Breakthrough ", 
        make_unique<Breakthrough>(), 
        SEARCH_DEPTH_BT, 
        NUM_SIMULATIONS
    );

    for (auto& rg : run_games) {
        cout << get<0>(rg)  << endl;
        test_performance(rg);
    }

    return 0;
}
