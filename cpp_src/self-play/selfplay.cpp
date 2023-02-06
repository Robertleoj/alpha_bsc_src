#include "../NN/nn.h"
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

    std::vector<nn::TrainingSample> samples;

    while(!game->is_terminal()){
        agent.search(100);

        auto visit_counts = agent.root_visit_counts();

        // std::cout << "Visit counts:" << std::endl;
        for(auto &p : visit_counts){
            std::cout << p.first << ": " << p.second << std::endl;
        }

    

        // std::cout << "Making policy tensor" << std::endl;
        auto policy_tensor =this->neural_net->visit_count_to_policy_tensor(visit_counts);

        // std::cout << "Making state tensor" << std::endl;
        auto state_tensor = this->neural_net->state_to_tensor(game->get_board());    Agent agent = Agent(game, pp::First, this->neural_net.get());


        nn::TrainingSample ts = {
            policy_tensor,
            state_tensor,
            0
        };

        // std::cout << "Made training sample" << std::endl;

        samples.push_back(ts);

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

    double outcome = agent.outcome_to_value(game->outcome(pp::First));

    for(auto &sample : samples){
        sample.outcome = outcome;
    }

    // now we need to insert the training data into the db
    this->db->insert_training_samples(samples);


    delete game;
}