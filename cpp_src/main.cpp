#include <iostream>
#include <random>
#include <memory>
#include <string>
#include "base/types.h"
#include "games/connect4.h"
#include "games/breakthrough.h"
#include "misc/perft.h"
#include "MCTS/agent.h"

using namespace std;
using namespace game;
using namespace games;

using RunGameEntry = tuple<string,unique_ptr<game::IGame>,int, int>;

void play_a_game(game::IGame& game)
{
    game.display(cout);
    cout << endl;
    auto agent = Agent(game, pp::First);
    while(!game.is_terminal()) {
        
        // for (auto it = ml->begin(), end = ml->end(); it != end; ++it) {
        //     cout << ' ' << game.move_as_str(it);
        // }
        

        // cout << endl;
        // std::uniform_int_distribution<int> distribution(0, ml->get_size()-1);
        // auto n = distribution(generator);
        //
        // game.make(ml->begin() + n);
        
        agent.get_move(1000);

        game.display(cout);
        
        cout << endl;
    }
    cout << str(game.outcome(First)) << endl;
}

int main()
{

    auto game = make_unique<Connect4>();
    play_a_game(*game.get());

    return 0;
}
