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
    
    bool agent_turn = true;

    while(!game.is_terminal()) {

        // if(agent_turn){
        auto mv = agent.get_move(10000);
        // cout << "Agent" << endl;
        cout << "Move :" << game.move_as_str(mv) << endl;
        // } else {
        //     auto ml = game.moves();
        //     int rand_idx = rand() % ml->get_size();
        //     auto mv = ml->begin() + rand_idx;
        //     game.make(mv);
        //     agent.update_tree(rand_idx);
        //     cout << "Move :" << game.move_as_str(mv) << endl;
        // }
        game.display(cout);
        
        agent_turn = !agent_turn;

        cout << endl;

        cout << endl;
    }
    cout << str(game.outcome(First)) << endl;
}

int main()
{
    srand(time(NULL));

    auto game = make_unique<Connect4>();
    play_a_game(*game.get());

    return 0;
}
