#include "../NN/connect4_nn.h"
#include "./selfplay.h"
#include "../utils/utils.h"
#include "../games/game.h"
#include "../games/connect4.h"
#include "../games/breakthrough.h"


SelfPlay::SelfPlay(std::string game) {
    this->game = game;

    this->db = std::make_unique<db::DB>(game);

    std::string model_path = utils::string_format(
        "../models/%s/%d.pt", 
        game.c_str(),
        this->db->curr_generation
    );

    if(game == "connect4"){
        this->neural_net = std::unique_ptr<nn::NN>(
            new nn::Connect4NN(model_path)
        );
    }

}


void SelfPlay::play_game(){

    game::IGame* game;

    if(this->game == "connect4"){
        game = new games::Connect4();
    } else if (this->game == "breakthrough") {
        game = new games::Breakthrough();
    }

    Agent agent = Agent(game, pp::First, this->neural_net.get());

    int num_moves = 0;

    game->display(std::cout);

    while(!game->is_terminal()){

        auto visit_counts = agent.search(100);

        // get best move id
        game::move_id best_move;
        int best_visit_count = -1;

        for(auto &p : visit_counts) {
            if(p.second > best_visit_count){
                best_move = p.first;
                best_visit_count = p.second;
            }
        }


        // hack to get the move iterator - make game accept move idx instead
        auto mv_idx = agent.tree->root->move_idx_of(best_move);
        auto mv = agent.tree->root->move_list->begin() + mv_idx;

        agent.update_tree(best_move);
        game->make(mv);


        game->display(std::cout);
        std::cout << std::endl;
        std::cout << "Move " << ++num_moves << std::endl;
        std::cout << std::endl;
    }

    std::cout << str(game->outcome(First)) << std::endl;

    delete game;
}