#include <iostream>
#include <random>

#include "./NN/nn.h"
#include "./NN/connect4_nn.h"
#include "./MCTS/agent.h"
#include "./config/config.h"
#include "./games/game.h"
#include "./games/connect4.h"
#include "./base/types.h"
#include "./utils/utils.h"





void play_user(){
    nn::NN * neural_net = new nn::Connect4NN("../models/connect4/18.pt");

    game::IGame* game = new games::Connect4();

    auto efunc = [neural_net](Board b){
        return std::move(neural_net->eval_state(b));
    };

    Agent agent(game, pp::First, efunc);

    bool agent_turn = true;

    game->display(std::cout);
    std::cout << std::endl;

    while(!game->is_terminal()){

        if(agent_turn){
            agent.search(config::hp["search_depth"].get<int>());

            auto visit_counts = agent.root_visit_counts();

            // get best move id

            game::move_id best_move;
            int best_visit_count = -1;

            for(auto &p : visit_counts) {
                if(p.second > best_visit_count){
                    best_move = p.first;
                    best_visit_count = p.second;
                }
            }

            std::string move_str = game->move_as_str(best_move);

            game->make(best_move);
            agent.update_tree(best_move);

            std::cout << "Agent made move "  << move_str << std::endl;
        } else {
            std::cout << "Move: ";
            int user_move;
            std::cin >> user_move;
            std::cout << std::endl << "You made move " << user_move << std::endl;
            game->make(user_move);
            agent.update_tree(user_move);
        }

        game->display(std::cout);
        std::cout << std::endl;

        agent_turn = !agent_turn;
    }
}

void print_pucts(Agent * ag) {
    std::cout << "PUCT, evaluations" << std::endl;
    MCNode * root = ag->tree->root;
    for(auto &mv: root->legal_moves){

        double v = -5;
        double p = -1;

        MCNode * child = root->children[mv];
        p = root->p_map[mv];

        if(child != nullptr){
            v = child->value_approx;
        }

        std::cout << mv 
                  << ": PUCT=" << ag->PUCT(ag->tree->root, mv) 
                  << ", v=" <<  v 
                  << ", p=" << p <<  std::endl;
    }
}

void play_self(std::string model1, std::string model2){
    nn::NN * neural_net = new nn::Connect4NN(model1);
    nn::NN * neural_net2 = new nn::Connect4NN(model2);

    game::IGame* game = new games::Connect4();

    auto efunc1 = [neural_net](Board b){
        return std::move(neural_net->eval_state(b));
    };

    auto efunc2 = [neural_net2](Board b){
        return std::move(neural_net2->eval_state(b));
    };

    Agent agent1(game, pp::First, efunc1, false);
    Agent agent2(game, pp::First, efunc2, false);

    bool agent1_turn = true;

    game->display(std::cout);
    std::cout << std::endl;


    while(!game->is_terminal()){
        
        std::map<game::move_id, int> visit_counts;
        if(agent1_turn){
            agent1.search(config::hp["search_depth"].get<int>());
            visit_counts = agent1.root_visit_counts();
            print_pucts(&agent1);
        } else {
            agent2.search(config::hp["search_depth"].get<int>());
            visit_counts = agent2.root_visit_counts();
            print_pucts(&agent2);
        }


            // get best move id

        game::move_id best_move;
        int best_visit_count = -1;
        // int best_visit_count = 100000000;

        for(auto &p : visit_counts) {
            if(p.second > best_visit_count){
            // if(p.second < best_visit_count){
                best_move = p.first;
                best_visit_count = p.second;
            }
        }

        std::string move_str = game->move_as_str(best_move);

        game->make(best_move);
        agent1.update_tree(best_move);
        agent2.update_tree(best_move);


        std::cout << "Agent" << (agent1_turn ? "1" : "2") << " made move "  << move_str << std::endl;

        game->display(std::cout);
        std::cout << std::endl;

        agent1_turn = !agent1_turn;
    }



}



int main(int argc, char * argv[]){

    std::string model1 = utils::string_format("../models/connect4/%s.pt", argv[1]);
    std::string model2 = utils::string_format("../models/connect4/%s.pt", argv[2]);

    config::initialize();
    srand(time(NULL));

    play_self(model1, model2);
}