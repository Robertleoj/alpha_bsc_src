#include "endgame_playouts.h"
#include "structs.h"
#include <string>
#include <unistd.h>
#include "simulation.h"
#include "../NN/connect4_nn.h"
#include "mutual.h"

db::TrainingSample get_training_sample(
    nn::move_dist normalized_visit_counts,
    std::string moves,
    Board board,
    nn::NN* neural_net,
    double weight
) {
    auto policy_tensor = neural_net->move_map_to_policy_tensor(normalized_visit_counts);
    auto state_tensor = neural_net->state_to_tensor(board);

    db::TrainingSample ts = {
        policy_tensor,
        state_tensor,
        0,
        board.to_move,
        moves,
        0,
        weight
    };

    return ts;
}

void endgame_update_training_sample(
    db::TrainingSample* sample, 
    nn::move_dist move_map,
    nn::NN * neural_net
){
    auto policy_tensor = neural_net->move_map_to_policy_tensor(move_map);
    sample->target_policy = policy_tensor;
    sample->weight = 1;
}

void write_samples(
    std::vector<db::TrainingSample>* training_samples,
    Agent* agent,
    ThreadData* thread_data,
    game::IGame* game,
    int num_moves
) {
    double outcome = agent->outcome_to_value(game->outcome(pp::First));

    int i = 0;
    for (auto& sample : *training_samples) {
        sample.outcome = agent->eval_for_player(outcome, sample.player);
        sample.moves_left = num_moves - i;
        i++;
    }

    // now we need to insert the training data into the db
    thread_data->db_mutex.lock();
    thread_data->db->insert_training_samples(training_samples);
    thread_data->db_mutex.unlock();
}

game::move_id select_move(nn::move_dist visit_count_dist, int num_moves) {

    int argmax_depth = config::hp["depth_until_pi_argmax"].get<int>();

    if (num_moves < argmax_depth) {
        return utils::sample_multinomial(visit_count_dist);
    }
    else {
        return utils::argmax_map(visit_count_dist);
    }
}

void thread_play(
    ThreadData* thread_data,
    std::string game_name
) {

    const int num_games = config::hp["games_per_thread"].get<int>();
    const int search_depth = config::hp["search_depth"].get<int>();

    Agent* agents[num_games];
    game::IGame* games[num_games];
    EvalRequest requests[num_games];
    bool dead_game[num_games];
    std::vector<db::TrainingSample> samples[num_games];
    std::stringstream moves[num_games];
    int num_moves[num_games];
    memset(num_moves, 0, sizeof(num_moves));
    bool on_backtrack_move[num_games];
    memset(on_backtrack_move, false, sizeof(on_backtrack_move));

    double current_gen = thread_data->db->curr_generation;

    EndgamePlayoutWeights endgame_playout_weights(current_gen);
    bool use_endgame_playout = config::hp["use_endgame_playouts"].get<bool>();
    int min_playout = config::hp["endgame_min_playouts"].get<int>();


    auto get_playouts = [search_depth, &endgame_playout_weights, use_endgame_playout, min_playout](int move){
        int playouts = search_depth;
        if(use_endgame_playout){
            double weight = endgame_playout_weights((double) move);
            playouts = (int) ((double ) search_depth * weight);
            if(playouts < min_playout){
                playouts = min_playout;
            }
        }

        return playouts;
    };

    auto get_weight = [&endgame_playout_weights, use_endgame_playout](int move){
        double weight = 1.0;
        if(use_endgame_playout){
            weight = endgame_playout_weights((double) move);
        }
        return weight;
    };

    // start the games
    for (int i = 0; i < num_games; i++) {
        thread_data->start_game_mutex.lock();
        if (thread_data->games_left <= 0) {
            thread_data->start_game_mutex.unlock();
            dead_game[i] = true;
            thread_data->num_active_games--;
            thread_data->q_cv.notify_all();
            continue;
        }

        thread_data->games_left--;
        thread_data->start_game_mutex.unlock();

        games[i] = get_game_instance(game_name);
        agents[i] = new Agent(games[i]);
        // thread_data->num_active_games++;
        dead_game[i] = false;

        auto [done, board] = agents[i]->init_mcts(get_playouts(0));

        queue_request(thread_data, board, &requests[i]);
    }

    while (true) {
        bool all_dead = true;
        for (int i = 0; i < num_games; i++) {

            if (!dead_game[i]) {
                all_dead = false;
            }

            if (dead_game[i] || !requests[i].completed) {
                usleep(0); // give other threads a chance to run
                continue;
            }
            // get answer
            std::unique_ptr<nn::NNOut> answer = std::move(requests[i].result);
            auto [done, board] = agents[i]->step(std::move(answer));

            while (done) {
                // agent is ready to move
                auto visit_counts = agents[i]->root_visit_counts();
                auto normalized_visit_counts = utils::softmax_map(visit_counts);

                double weight = get_weight(num_moves[i]);

                if(on_backtrack_move[i]){
                    weight = 1;
                }

                if(!on_backtrack_move[i]){
                    samples[i].push_back(get_training_sample(
                        normalized_visit_counts,
                        moves[i].str(),
                        games[i]->get_board(),
                        thread_data->neural_net,
                        weight
                    ));
                }

                auto best_move = select_move(normalized_visit_counts, num_moves[i]);

                moves[i] << games[i]->move_as_str(best_move) << ";";

                games[i]->make(best_move);
                agents[i]->update_tree(best_move);

                num_moves[i]++;
                // std::cout << "Made move " << num_moves[i] << std::endl;

                bool game_completed = games[i]->is_terminal();

                if (game_completed || on_backtrack_move[i]) {
                    if(!on_backtrack_move[i] && use_endgame_playout){
                        delete agents[i];
                        // samples[i].pop_back();
                        games[i]->push();
                        games[i]->retract(best_move);
                        agents[i] = new Agent(games[i]);
                        on_backtrack_move[i] = true;
                        num_moves[i]--;
                    } else{
                        if(on_backtrack_move[i]){
                            games[i]->pop();
                            endgame_update_training_sample(&samples[i].back(), normalized_visit_counts, thread_data->neural_net);
                        }

                        write_samples(&samples[i], agents[i], thread_data, games[i], num_moves[i]);

                        // restart game
                        samples[i].clear();
                        num_moves[i] = 0;
                        delete agents[i];
                        delete games[i];
                        std::cout << "Games left: " << thread_data->games_left + thread_data->num_active_games << std::endl;


                        thread_data->start_game_mutex.lock();
                        if (thread_data->games_left <= 0) {
                            thread_data->start_game_mutex.unlock();

                            dead_game[i] = true;
                            thread_data->num_active_games--;
                            thread_data->q_cv.notify_all();

                            goto cnt; // this is a goto, but it's the only way to break out of two loops
                        }
                        thread_data->games_left--;
                        thread_data->start_game_mutex.unlock();

                        games[i] = get_game_instance(game_name);
                        agents[i] = new Agent(games[i]);
                        on_backtrack_move[i] = false;
                    }
                }
                
                int playouts = get_playouts(num_moves[i]);
                if(on_backtrack_move[i]){
                    playouts = search_depth;
                }
                std::tie(done, board) = agents[i]->init_mcts(playouts);
            }

            queue_request(thread_data, board, &requests[i]);

        cnt:;
        }
        if (all_dead) {
            break;
        }
    }

    std::cout << "Game thread done" << std::endl;
}

void self_play_start_threads(std::vector<std::thread>& threads, ThreadData* thread_data, int num_threads, std::string game_name) {
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(
            &thread_play, thread_data, game_name
        ));
    }
}

void sim::self_play(std::string game) {

    int num_threads = config::hp["self_play_num_threads"].get<int>();
    int num_games = config::hp["self_play_num_games"].get<int>();

    std::vector<std::thread> threads;

    auto thread_data = init_thread_data(game, num_games);

    std::vector<std::thread> dl_threads;
    std::vector<std::thread> nn_threads;
    std::vector<std::thread> return_threads;

    BatchData batch_data;

    start_batching_threads(thread_data, &batch_data, dl_threads, nn_threads, return_threads);

    self_play_start_threads(threads, thread_data, num_threads, game);

    join_threads({ &dl_threads, &nn_threads, &return_threads, &threads });
}