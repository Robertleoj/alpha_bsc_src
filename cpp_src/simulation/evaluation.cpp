#include "./simulation.h"
#include "../utils/utils.h"
#include "../global.h"
#include <unistd.h>
#include "structs.h"
#include "mutual.h"

/**
 * @brief Load data from json file and make a queue of ground truth requests
 *
 * @param ground_truth_filename
 * @return std::queue<GroundTruthRequest>
 */
std::queue<GroundTruthRequest> make_eval_data_queue(std::string ground_truth_filename) {
    // read in the ground truth file - it is a json file
    std::ifstream ground_truth_file(ground_truth_filename);
    std::cout << "Reading ground truth file: " << ground_truth_filename << std::flush;

    nlohmann::json ground_truth_json = nlohmann::json::parse(ground_truth_file);

    std::cout << colors::GREEN << " [DONE]" << colors::RESET << std::endl;

    std::queue<GroundTruthRequest> eval_data_queue;

    for (auto& game : ground_truth_json.get<std::vector<nlohmann::json>>()) {
        std::string moves = game["moves"].get<std::string>();
        std::vector<double> ground_truth = game["target"].get<std::vector<double>>();
        double value = game["value"].get<double>();
        eval_data_queue.push(GroundTruthRequest{ moves, ground_truth, value });
    }
    return eval_data_queue;
}

std::vector<double> get_policy_vector(nn::move_dist prior_map) {
    std::vector<double> out;
    for (game::move_id move = 1; move <= 7; move++) {
        out.push_back(prior_map[move]);
    }

    return out;
}

double get_ce_loss(std::vector<double> vec1, std::vector<double> vec2) {
    auto cross_entropy = torch::nn::CrossEntropyLoss();
    return cross_entropy(
        torch::tensor(vec1),
        torch::tensor(vec2)
    ).item<double>();
}


void write_evaluation_to_db(
    ThreadData* thread_data,
    GroundTruthRequest* gt,
    std::vector<double>& policy_prior,
    double nn_value,
    std::vector<double>& policy_mcts,
    double mcts_value
) {

    // calculate errors
    double policy_prior_error = get_ce_loss(policy_prior, gt->ground_truth);
    double policy_mcts_error = get_ce_loss(policy_mcts, gt->ground_truth);

    double nn_value_error = std::pow(nn_value - gt->value, 2);
    double mcts_value_error = std::pow(mcts_value - gt->value, 2);

    // print some info
    std::cout << '.' << std::flush;
    // printf("Eval: %s\n", gt->moves.c_str());

    // create eval entry
    db::EvalEntry eval_entry {
        gt->moves,
            config::hp["eval_search_depth"].get<int>(),
            gt->ground_truth,
            gt->value,
            policy_prior,
            policy_prior_error,
            policy_mcts,
            policy_mcts_error,
            nn_value,
            nn_value_error,
            mcts_value,
            mcts_value_error
    };

    // write to db
    thread_data->db_mutex.lock();
    thread_data->db->insert_evaluation(&eval_entry);
    thread_data->db_mutex.unlock();
}

void init_eval_games(ThreadEvalData* data, ThreadData* thread_data, EvalData* eval_data, int num_games) {
    // start the games
    for (int i = 0; i < num_games; i++) {
        auto& ed = data[i];

        if (DEBUG) {
            std::cout << "Starting game " << i << std::endl;
        }

        eval_data->board_queue_mutex.lock();

        if (eval_data->board_queue.empty()) {
            eval_data->board_queue_mutex.unlock();
            ed.dead_game = true;
            thread_data->num_active_games--;
            thread_data->q_cv.notify_all();
            continue;
        }

        auto req = eval_data->board_queue.front();
        eval_data->board_queue.pop();
        eval_data->board_queue_mutex.unlock();
        ed.gt_request = req;

        ed.game = get_game_instance("connect4");

        for (char mv : req.moves) {
            ed.game->make(mv - '0');
        }

        ed.agent = new Agent(ed.game, false);

        ed.dead_game = false;

        auto [done, board] = ed.agent->init_mcts(config::hp["eval_search_depth"].get<int>());

        queue_request(thread_data, board, &ed.request);
    }
}

/**
 * @brief Plays games on separate threads until all games are completed
 *
 * @param thread_idx
 * @param eval_data
 */
void thread_eval(
    ThreadData* thread_data,
    EvalData* eval_data
) {
    if (DEBUG) {
        std::cout << "Started eval thread" << std::endl;
    }

    const int num_games = config::hp["games_per_thread"].get<int>();

    ThreadEvalData data[num_games];

    init_eval_games(data, thread_data, eval_data, num_games);

    if (DEBUG) {
        std::cout << "Started games" << std::endl;
    }

    while (true) {
        bool all_dead = true;
        for (int i = 0; i < num_games; i++) {
            auto& ed = data[i];

            if (!ed.dead_game) {
                all_dead = false;
            }

            if (ed.dead_game || !ed.request.completed) {
                usleep(0); // give other threads a chance to run
                continue;
            }

            // get answer
            if (DEBUG) {
                std::cout << "Got answer" << std::endl;
            }

            std::unique_ptr<nn::NNOut> answer = std::move(ed.request.result);

            if (!ed.first_nn_out_set) {
                ed.first_nn_out = *answer;
                ed.first_nn_out_set = true;
            }

            auto [done, board] = ed.agent->step(std::move(answer));

            if (done) {
                if (DEBUG) {
                    std::cout << "Evaluated board" << std::endl;
                }

                auto visit_counts = ed.agent->root_visit_counts();
                auto pol_mcts = utils::softmax_map(visit_counts);
                std::vector<double> pol_mcts_vec = get_policy_vector(pol_mcts);


                auto pol_prior = ed.first_nn_out.p;
                std::vector<double> pol_prior_vec = get_policy_vector(pol_prior);

                double nn_value = ed.first_nn_out.v;

                double mcts_value = ed.agent->tree->root->value_approx;

                write_evaluation_to_db(
                    thread_data,
                    &ed.gt_request,
                    pol_prior_vec,
                    nn_value,
                    pol_mcts_vec,
                    mcts_value
                );

                // restart game
                delete ed.agent;
                delete ed.game;


                eval_data->board_queue_mutex.lock();

                if (eval_data->board_queue.empty()) {
                    eval_data->board_queue_mutex.unlock();
                    ed.dead_game = true;
                    thread_data->num_active_games--;
                    thread_data->q_cv.notify_all();
                    continue;

                }
                else {
                    // start new game
                    auto req = eval_data->board_queue.front();
                    eval_data->board_queue.pop();
                    eval_data->board_queue_mutex.unlock();

                    ed.gt_request = req;
                    ed.first_nn_out_set = false;

                    ed.game = get_game_instance("connect4");

                    for (char mv : req.moves) {
                        ed.game->make(mv - '0');
                    }

                    ed.agent = new Agent(ed.game, false);

                    std::tie(done, board) = ed.agent->init_mcts(config::hp["eval_search_depth"].get<int>());
                }
            }
            queue_request(thread_data, board, &ed.request);
        }
        if (all_dead) {
            break;
        }
    }

    std::cout << "Eval thread done" << std::endl;
}



void start_eval_threads(
    ThreadData* thread_data,
    EvalData* eval_data,
    std::vector<std::thread>& threads,
    int num_threads
) {
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(thread_eval, thread_data, eval_data));
    }
}

void sim::eval_targets(std::string eval_targets_filename, int generation_num) {

    int num_threads = config::hp["self_play_num_threads"].get<int>();
    int num_games = config::hp["self_play_num_games"].get<int>();

    EvalData eval_data;

    auto eval_data_queue = make_eval_data_queue(eval_targets_filename);
    eval_data.board_queue = eval_data_queue;

    std::vector<std::thread> threads;

    auto thread_data = init_thread_data("connect4", num_games, generation_num);

    std::vector<std::thread> dl_threads;
    std::vector<std::thread> nn_threads;
    std::vector<std::thread> return_threads;

    BatchData batch_data;

    if (DEBUG) {
        std::cout << "Starting batching threads" << std::endl;
    }

    start_batching_threads(thread_data, &batch_data, dl_threads, nn_threads, return_threads);

    // self_play_start_threads(threads, thread_data, num_threads, game);

    if (DEBUG) {
        std::cout << "Starting eval threads" << std::endl;
    }

    start_eval_threads(thread_data, &eval_data, threads, num_threads);

    join_threads({ &dl_threads, &nn_threads, &return_threads, &threads });

}
