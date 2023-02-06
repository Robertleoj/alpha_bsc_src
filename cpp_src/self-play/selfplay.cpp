#include "../NN/nn.h"
#include "../NN/connect4_nn.h"
#include "./selfplay.h"
#include "../utils/utils.h"
#include "../games/game.h"
#include "../games/connect4.h"
#include "../games/breakthrough.h"
#include "../hyperparams.h"
#include "../base/types.h"
#include <thread>


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

void SelfPlay::self_play(){
    int num_threads = hp::num_parallel_games;

    std::mutex queue_mutex;
    std::mutex db_mutex;

    std::queue<eval_request> eval_requests;

    std::thread threads[num_threads];

    bool request_completed[num_threads];

    memset(request_completed, false, sizeof(request_completed));

    std::mutex req_comp_mutex;

    std::unique_ptr<nn::NNOut> evaluations[num_threads];

    std::condition_variable eval_cv;

    std::condition_variable nn_q_wait_cv;

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
            &nn_q_wait_cv
        );
    }

    while(1){
        std::unique_lock<std::mutex> nn_q_lock(queue_mutex);
        nn_q_wait_cv.wait(nn_q_lock, [eval_requests](){
            return eval_requests.size() >= hp::batch_size;
        });
        std::vector<int> thread_indices;
        std::vector<Board> states;

        for(int i = 0; i < hp::batch_size; i++){
            auto p = eval_requests.front();
            thread_indices.push_back(p.first);
            states.push_back(p.second);
            eval_requests.pop();
        }

        nn_q_lock.unlock();

        auto result = this->neural_net->eval_states(&states);
        for(int i = 0; i < thread_indices.size(); i++){
            int thread_idx = thread_indices[i];
            evaluations[thread_idx] = std::move(result[i]);

            // req_comp_mutex.lock();
            request_completed[thread_idx] = true;
            // req_comp_mutex.unlock();
        }

        eval_cv.notify_all();
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
    std::condition_variable * nn_q_wait_cv
){
    game::IGame* game;

    if(this->game == "connect4"){
        game = new games::Connect4();
    } else if (this->game == "breakthrough") {
        game = new games::Breakthrough();
    }   

    // evaluation function for agent
    eval_f eval_func = [
        thread_idx, 
        q_mutex, 
        q,
        req_completed,
        req_comp_mutex,
        evaluations,
        eval_cv,
        nn_q_wait_cv
    ](Board b){
        
        // Put item in queue
        q_mutex->lock();
        q->push({thread_idx, b});
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
        return std::move(evaluations[thread_idx]);
    };
    
    Agent agent = Agent(game, pp::First, eval_func);

    int num_moves = 0;

    std::vector<nn::TrainingSample> samples;

    while(!game->is_terminal()){
        agent.search(hp::search_depth);

        auto visit_counts = agent.root_visit_counts();

        // std::cout << "Making policy tensor" << std::endl;
        auto policy_tensor =this->neural_net->visit_count_to_policy_tensor(visit_counts);

        // std::cout << "Making state tensor" << std::endl;
        auto state_tensor = this->neural_net->state_to_tensor(game->get_board());    


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


    }

    double outcome = agent.outcome_to_value(game->outcome(pp::First));

    for(auto &sample : samples){
        sample.outcome = outcome;
    }

    // now we need to insert the training data into the db
    db_mutex->lock();
    this->db->insert_training_samples(samples);
    db_mutex->unlock();

    delete game;
}


// void SelfPlay::play_game(){

//     game::IGame* game;

//     if(this->game == "connect4"){
//         game = new games::Connect4();
//     } else if (this->game == "breakthrough") {
//         game = new games::Breakthrough();
//     }

//     Agent agent = Agent(game, pp::First, this->neural_net.get());

//     int num_moves = 0;

//     game->display(std::cout);

//     std::vector<nn::TrainingSample> samples;

//     while(!game->is_terminal()){
//         agent.search(100);

//         auto visit_counts = agent.root_visit_counts();

//         // std::cout << "Visit counts:" << std::endl;
//         for(auto &p : visit_counts){
//             std::cout << p.first << ": " << p.second << std::endl;
//         }

    

//         // std::cout << "Making policy tensor" << std::endl;
//         auto policy_tensor =this->neural_net->visit_count_to_policy_tensor(visit_counts);

//         // std::cout << "Making state tensor" << std::endl;
//         auto state_tensor = this->neural_net->state_to_tensor(game->get_board());    


//         nn::TrainingSample ts = {
//             policy_tensor,
//             state_tensor,
//             0
//         };

//         // std::cout << "Made training sample" << std::endl;

//         samples.push_back(ts);

//         // get best move id
//         game::move_id best_move;
//         int best_visit_count = -1;


//         for(auto &p : visit_counts) {
//             if(p.second > best_visit_count){
//                 best_move = p.first;
//                 best_visit_count = p.second;
//             }
//         }

//         // hack to get the move iterator - make game accept move idx instead
//         auto mv_idx = agent.tree->root->move_idx_of(best_move);
//         auto mv = agent.tree->root->move_list->begin() + mv_idx;

//         agent.update_tree(best_move);
//         game->make(mv);


//         game->display(std::cout);
//         std::cout << std::endl;
//         std::cout << "Move " << ++num_moves << std::endl;
//         std::cout << std::endl;
//     }
//     std::cout << str(game->outcome(First)) << std::endl;

//     double outcome = agent.outcome_to_value(game->outcome(pp::First));

//     for(auto &sample : samples){
//         sample.outcome = outcome;
//     }

//     // now we need to insert the training data into the db
//     this->db->insert_training_samples(samples);


//     delete game;
// }