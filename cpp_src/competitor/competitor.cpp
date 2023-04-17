#include <filesystem>
#include "../utils/utils.h"
#include "./competitor.h"
#include "../simulation/mutual.h"
#include "../config/config.h"



Competitor::Competitor(
    std::string run_name, 
    std::string game_name,
    int generation,
    int num_agents,
    int num_playouts
) {
    this->num_playouts = num_playouts;
    this->num_dead = 0;
    std::filesystem::path curr_path = std::filesystem::current_path();

    // cd into right directory
    std::filesystem::path run_path = curr_path.parent_path() / "vault" / game_name / run_name;
    std::filesystem::current_path(run_path);
    config::initialize();

    this->results = std::vector(num_agents, 0.0);
    this->dead = std::vector(num_agents, false);

    for(int i = 0; i < num_agents; i++) {
        this->games.push_back(game_name == "connect4" ? (game::IGame *)(new games::Connect4()) : (game::IGame *)(new games::Breakthrough()));
        this->agents.push_back(new Agent(this->games[i],false));
    }

    std::filesystem::path nn_path = std::filesystem::current_path() / "models" / utils::string_format("%d.pt", generation);

    if (game_name == "connect4") {
        this->neural_net = new nn::Connect4NN(nn_path);
    } else {
        this->neural_net = new nn::BreakthroughNN(nn_path);
    }

    std::filesystem::current_path(curr_path);
}

void Competitor::update(std::vector<std::string> moves) {
    for(int i = 0; i < this->games.size(); i++) {
        if(moves[i] == "") {
            continue;
        }

        if (!this->dead[i]){
            game::move_id move_id = mm::from_str(moves[i]);
            this->games[i]->make(move_id);

            if(this->games[i]->is_terminal()) {

                auto outcome = this->games[i]->outcome(
                    this->games[i]->get_to_move()
                );

                this->results[i] = this->agents[i]->outcome_to_value(outcome);

                this->dead[i] = true;
                delete this->agents[i];
                delete this->games[i];
                this->num_dead++;
            } else {
                this->agents[i]->update_tree(move_id);
            }
        } else {
            throw std::runtime_error("Agent is dead");
        }
    }
}


std::vector<std::string> Competitor::make_and_get_moves() {  
    std::vector<std::string> moves(this->games.size());

    int moves_left = this->games.size() - this->num_dead;

    std::vector<std::unique_ptr<nn::NNOut>> last_answers(this->games.size());
    std::vector<bool> started(this->games.size(), false);

    while(moves_left > 0) {
        std::vector<int> agent_indices;
        std::vector<std::vector<game::move_id>> legal_moves;
        std::vector<pp::Player> to_move;
        std::vector<at::Tensor> states;

        for(int i = 0; i < this->games.size(); i++) {
            if(!this->dead[i] && moves[i] == "") {
                bool done;
                Board board;

                if(!started[i]) {
                    started[i] = true;
                    // init mcts
                    std::tie(done, board) = this->agents[i]->init_mcts(this->num_playouts);
                } else {
                    // step agents
                    std::tie(done, board) = this->agents[i]->step(std::move(last_answers[i]));
                }
                if(done) {
                    game::move_id best_move = this->agents[i]->best_move();
                    this->games[i]->make(best_move);
                    moves[i] = this->games[i]->move_as_str(best_move);

                    if(this->games[i]->is_terminal()) {
                        auto player = this->games[i]->get_to_move();
                        auto outcome = this->games[i]->outcome(player);
                        this->results[i] = this->agents[i]->switch_eval(this->agents[i]->outcome_to_value(outcome));

                        delete this->agents[i];
                        delete this->games[i];
                        this->num_dead++;
                        this->dead[i] = true;
                    } else {
                        this->agents[i]->update_tree(best_move);
                    }
                    moves_left--;
                } else {
                    agent_indices.push_back(i);
                    legal_moves.push_back(games[i]->moves());
                    states.push_back(this->neural_net->state_to_tensor(board));
                    to_move.push_back(board.to_move);
                }
            }
        }

        if(states.size() > 0) {
            at::Tensor batch = this->neural_net->prepare_batch(states);
            auto batch_result = this->neural_net->run_batch(batch);

            auto pol_tensor = batch_result.at(0).toTensor().cpu();
            auto val_tensor = batch_result.at(1).toTensor().cpu();

            std::vector<std::vector<game::move_id>*> legal_moves_ptrs;
            std::transform(legal_moves.begin(), legal_moves.end(), std::back_inserter(legal_moves_ptrs), [](auto &v){return &v;});


            auto nnouts = this->neural_net->net_out_to_nnout(
                pol_tensor,
                val_tensor,
                legal_moves_ptrs,
                &to_move
            );

            for(int i = 0; i < agent_indices.size(); i++) {
                last_answers[agent_indices[i]] = std::move(nnouts[i]);
            }
        }
    }

    return moves;
}

std::vector<double> Competitor::get_results() {
    return this->results;
}