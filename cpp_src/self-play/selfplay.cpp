#include "../NN/nn.h"
#include "../NN/connect4_nn.h"
#include "./selfplay.h"
#include "../utils/utils.h"
#include "../games/game.h"
#include "../games/connect4.h"
#include "../games/breakthrough.h"
#include "../config/config.h"
#include "../base/types.h"
#include "../utils/utils.h"
#include <thread>


SelfPlay::SelfPlay(std::string game) {
    this->game = game;

    this->db = std::make_unique<db::DB>(game);

    std::string model_path = utils::string_format(
        "../models/%s/%d.pt", 
        game.c_str(),
        this->db->curr_generation
    );

    std::cout << "making neural net" << std::endl;
    if(game == "connect4"){
        this->neural_net = std::unique_ptr<nn::NN>(
            new nn::Connect4NN(model_path)
        );
    }
    std::cout << "made neural net" << std::endl;
}

void SelfPlay::start_threads(std::thread *threads, ThreadData *thread_data, int num_threads)
{
    for(int i = 0; i < num_threads; i++){
        threads[i] = std::thread(
            &SelfPlay::thread_play, this, i, thread_data
        );
    }
}

/**
 * @brief Starts the self play process
 * 
 */
void SelfPlay::self_play(){
    int num_threads = config::hp["num_parallel_games"].get<int>();


    std::thread threads[num_threads];

    // std::atomic<int> games_left(hp::self_play_num_games);
    int num_games = config::hp["self_play_num_games"].get<int>();

    bool req_completed[num_threads];
    memset(req_completed, false, sizeof(req_completed));


    std::unique_ptr<nn::NNOut> evaluations[num_threads];

    ThreadData thread_data(
        num_threads, 
        num_games, 
        (bool *)req_completed, 
        (std::unique_ptr<nn::NNOut> *)evaluations
    );

    this->start_threads(threads, &thread_data, num_threads);


    auto bs = [&thread_data](){
        return std::min(
            (int)thread_data.num_active_threads, 
            config::hp["batch_size"].get<int>()
        );
    };

    while(thread_data.num_active_threads > 0){
        std::unique_lock<std::mutex> nn_q_lock(thread_data.q_mutex);

        thread_data.q_cv.wait(nn_q_lock, [&thread_data, &bs](){
            return thread_data.eval_q.size() >= bs();
        });

        if(bs() == 0){
            nn_q_lock.unlock();
            break;
        }

        std::vector<int> thread_indices;
        std::vector<at::Tensor> states;

        for(int i = 0; i < bs(); i++){
            auto p = thread_data.eval_q.front();
            thread_data.eval_q.pop();

            thread_indices.push_back(p.first);
            states.push_back(p.second);
        }

        nn_q_lock.unlock();

        // std::cout << "Running neural network on " << states.size() << " inputs" << std::endl;

        auto result = this->neural_net->eval_tensors(states);
        for(int i = 0; i < (int)thread_indices.size(); i++){
            int thread_idx = thread_indices[i];

            evaluations[thread_idx] = std::move(result[i]);

            thread_data.req_completed[thread_idx] = true;
        }

        thread_data.eval_cv.notify_all();
    }

    for(auto &t: threads){
        t.join();
    }
}

/**
 * @brief Plays games on separate threads until all games are completed
 * 
 * 
 * 
 * @param thread_idx 
 * @param thread_data
 */
void SelfPlay::thread_play(
    int thread_idx,
    ThreadData * thread_data
) {

    while((thread_data->games_left) > 0){
        (thread_data->games_left)--;

        game::IGame* game;

        if(this->game == "connect4"){
            game = new games::Connect4();
        } else if (this->game == "breakthrough") {
            game = new games::Breakthrough();
        }   

        nn::NN * nn_ptr = this->neural_net.get();
        // evaluation function for agent
        eval_f eval_func = [
            thread_idx, 
            &thread_data,
            nn_ptr
        ](Board b){
            
            auto t = nn_ptr->state_to_tensor(b);
            // Put item in queue
            thread_data->q_mutex.lock();
            thread_data->eval_q.push({thread_idx, t});
            thread_data->q_mutex.unlock();

            thread_data->q_cv.notify_one();

            // Wait for result
            std::unique_lock<std::mutex> lq(thread_data->req_completed_mutex);

            thread_data->eval_cv.wait(lq, [thread_data, thread_idx](){
                return thread_data->req_completed[thread_idx];
            });
            
            thread_data->req_completed[thread_idx] = false;
            lq.unlock();

            // Get results
            thread_data->results_mutex.lock();
            auto result = std::move(thread_data->evaluations[thread_idx]);
            thread_data->evaluations[thread_idx] = nullptr;
            thread_data->results_mutex.unlock();
            return result;
        };
        
        Agent * agent = new Agent(game, eval_func);

        int num_moves = 0;

        std::vector<nn::TrainingSample> samples;

        // game->display(std::cout);

        std::stringstream moves;

        int argmax_depth = config::hp["depth_until_pi_argmax"].get<int>();

        while(!game->is_terminal()){

            agent->search(
                config::hp["search_depth"].get<int>()
            );

            auto visit_counts = agent->root_visit_counts();

            // std::cout << "Making policy tensor" << std::endl;
            std::map<game::move_id, double> normalized_visit_counts = utils::softmax_map(visit_counts);

            auto policy_tensor =this->neural_net->move_map_to_policy_tensor(normalized_visit_counts);
            // std::cout << "made policy tensor" << std::endl;

            // std::cout << "Making state tensor" << std::endl;
            auto state_tensor = this->neural_net->state_to_tensor(game->get_board());    
            // std::cout << "made state tensor" << std::endl;

            // get best move id
            game::move_id best_move;

            nn::TrainingSample ts = {
                policy_tensor,
                state_tensor,
                0,
                game->get_to_move(),
                moves.str()
            };

            // std::cout << "Made training sample" << std::endl;

            samples.push_back(ts);

            if(num_moves < argmax_depth){
                best_move = utils::sample_multinomial(normalized_visit_counts);
            } else {
                best_move = utils::argmax_map(visit_counts);
            }

            std::string move_str = game->move_as_str(best_move);
            moves << move_str << ";";

            game->make(best_move);
            num_moves++;

            agent->update_tree(best_move);
        }

        std::cout << "finished game" << std::endl;

        double outcome = agent->outcome_to_value(game->outcome(pp::First));

        for(auto &sample : samples){
            sample.outcome = agent->eval_for_player(outcome, sample.player);
        }

        // now we need to insert the training data into the db
        thread_data->db_mutex.lock();
        // db::DB db(this->game);
        this->db->insert_training_samples(samples);
        thread_data->db_mutex.unlock();

        delete agent;
        delete game;
        std::cout << "Games left: " << thread_data->games_left + thread_data->num_active_threads << std::endl;
    }
    thread_data->q_cv.notify_one();
    // std::cout << "Active threads left: " << at << std::endl;
}
