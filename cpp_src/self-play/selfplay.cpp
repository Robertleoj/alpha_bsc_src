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

void SelfPlay::self_play(){
    int num_threads = config::hp["num_parallel_games"].get<int>();

    std::mutex queue_mutex;
    std::mutex db_mutex;

    std::queue<eval_request> eval_requests;

    std::thread threads[num_threads];

    bool request_completed[num_threads];

    memset(request_completed, false, sizeof(request_completed));

    std::mutex req_comp_mutex;
    std::mutex results_mutex;

    std::unique_ptr<nn::NNOut> evaluations[num_threads];

    std::condition_variable eval_cv;

    std::condition_variable nn_q_wait_cv;

    // std::atomic<int> games_left(hp::self_play_num_games);
    std::atomic<int> games_left(
        config::hp["self_play_num_games"].get<int>()
    );
    std::atomic<int> num_active_threads(num_threads);

    for(int i = 0; i < num_threads; i++){
        threads[i] = std::thread(
            &SelfPlay::thread_play, this,
            i, 
            &eval_requests,
            &queue_mutex,
            &db_mutex,
            (bool *) request_completed,
            &req_comp_mutex,
            (std::unique_ptr<nn::NNOut> *) evaluations,
            &eval_cv,
            &nn_q_wait_cv,
            &results_mutex,
            &games_left,
            &num_active_threads
        );
    }

    auto bs = [&num_active_threads](){
        return std::min(
            (int)num_active_threads, 
            config::hp["batch_size"].get<int>()
        );
    };

    while(num_active_threads > 0){
        std::unique_lock<std::mutex> nn_q_lock(queue_mutex);
        nn_q_wait_cv.wait(nn_q_lock, [&eval_requests, &num_active_threads, &bs](){
            return eval_requests.size() >= bs();
        });

        if(bs() == 0){
            nn_q_lock.unlock();
            break;
        }

        std::vector<int> thread_indices;
        std::vector<at::Tensor> states;

        for(int i = 0; i < bs(); i++){
            auto p = eval_requests.front();
            thread_indices.push_back(p.first);
            states.push_back(p.second);
            eval_requests.pop();
        }

        nn_q_lock.unlock();

        // std::cout << "Running neural network on " << states.size() << " inputs" << std::endl;

        auto result = this->neural_net->eval_tensors(states);
        for(int i = 0; i < (int)thread_indices.size(); i++){
            int thread_idx = thread_indices[i];

            evaluations[thread_idx] = std::move(result[i]);

            request_completed[thread_idx] = true;
        }

        eval_cv.notify_all();
    }

    for(auto &t: threads){
        t.join();
    }
}

void SelfPlay::thread_play(
    int thread_idx, 
    std::queue<eval_request>* q,
    std::mutex * q_mutex,
    std::mutex * db_mutex,
    bool * req_completed,
    std::mutex * req_comp_mutex,
    std::unique_ptr<nn::NNOut> * evaluations,
    std::condition_variable * eval_cv,
    std::condition_variable * nn_q_wait_cv,
    std::mutex * results_mutex,
    std::atomic<int> * games_left,
    std::atomic<int> * num_active_threads
) {

    while((*games_left) > 0){
        (*games_left)--;

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
            q_mutex, 
            q,
            req_completed,
            req_comp_mutex,
            evaluations,
            eval_cv,
            nn_q_wait_cv,
            results_mutex,
            nn_ptr
        ](Board b){
            
            auto t = nn_ptr->state_to_tensor(b);
            // Put item in queue
            q_mutex->lock();
            q->push({thread_idx, t});
            q_mutex->unlock();

            nn_q_wait_cv->notify_one();

            // Wait for result
            std::unique_lock<std::mutex> lq(*req_comp_mutex);
            eval_cv->wait(lq, [req_completed, thread_idx](){
                return req_completed[thread_idx];
            });
            
            req_completed[thread_idx] = false;
            lq.unlock();

            // Get results
            results_mutex->lock();
            auto result = std::move(evaluations[thread_idx]);
            evaluations[thread_idx] = nullptr;
            results_mutex->unlock();
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
            int best_visit_count = -1;

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
        db_mutex->lock();
        // db::DB db(this->game);
        this->db->insert_training_samples(samples);
        db_mutex->unlock();

        delete agent;
        delete game;
        std::cout << "Games left: " << *games_left + *num_active_threads << std::endl;
    }
    int at = --(*num_active_threads);
    nn_q_wait_cv->notify_one();
    // std::cout << "Active threads left: " << at << std::endl;
}
