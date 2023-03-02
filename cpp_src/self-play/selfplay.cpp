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
#include <stdexcept>
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
 * @brief Get the queue from thread_data, pops from it and adds to states and thread_indices.
 *
 * Gets a reference to thead_data, states and thread_indices.
 * 
 * @param thread_data 
 * @param bs 
 * @param thread_indices 
 * @param states 
 * @return std::pair<std::vector<int>, std::vector<at::Tensor>> 
 */
void SelfPlay::pop_batch(
    ThreadData &thread_data, 
    std::function<int()> bs, 
    std::vector<int> &thread_indices, 
    std::vector<at::Tensor> &states
    )
{
    for(int i = 0; i < bs(); i++){
        auto p = thread_data.eval_q.front();
        thread_data.eval_q.pop();

        thread_indices.push_back(p.first);
        states.push_back(p.second);
    }
}

void SelfPlay::eval_batch(std::vector<at::Tensor> &states, std::vector<int> &thread_indices, ThreadData &thread_data)
{
    auto result = this->neural_net->eval_tensors(states);
        for(int i = 0; i < (int)thread_indices.size(); i++){
            int thread_idx = thread_indices[i];

            thread_data.evaluations[thread_idx] = std::move(result[i]);

            thread_data.req_completed[thread_idx] = true;
        }
        thread_data.eval_cv.notify_all();
}

void SelfPlay::active_thread_work(ThreadData &thread_data, std::function<unsigned long()> bs)
{
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

        this->pop_batch(thread_data, bs, thread_indices, states);

        nn_q_lock.unlock();

        this->eval_batch(states, thread_indices, thread_data);
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
        return (unsigned long)std::min(
            (int)thread_data.num_active_threads, 
            config::hp["batch_size"].get<int>()
        );
    };

    this->active_thread_work(thread_data, bs);

    for(auto &t: threads){
        t.join();
    }
}

/**
 * @brief Creates an eval function for a thread
 * 
 * @param thread_idx 
 * @param thread_data 
 * @param nn_ptr 
 * @return eval_f 
 */
eval_f SelfPlay::make_eval_function(int thread_idx, ThreadData * thread_data, nn::NN * nn_ptr){
    return [
        thread_idx, 
        thread_data,
        nn_ptr
    ](Board b){
        
        // Make thread create the tensor from state so main thread does not have to
        auto t = nn_ptr->state_to_tensor(b);

        // Put item in queue
        thread_data->q_mutex.lock();
        thread_data->eval_q.push({thread_idx, t});
        thread_data->q_mutex.unlock();

        // notify main thread that there is an item in the queue
        thread_data->q_cv.notify_one();

        // Wait for result
        std::unique_lock<std::mutex> lq(thread_data->req_completed_mutex);

        // Wait for main thread to notify that the result is ready
        thread_data->eval_cv.wait(lq, [thread_data, thread_idx](){
            return thread_data->req_completed[thread_idx];
        });
        
        // Reset flag
        thread_data->req_completed[thread_idx] = false;
        lq.unlock();

        // Get results
        thread_data->results_mutex.lock();
        auto result = std::move(thread_data->evaluations[thread_idx]);
        thread_data->evaluations[thread_idx] = nullptr;
        thread_data->results_mutex.unlock();

        return result;
    };   
}

game::IGame * SelfPlay::get_game_instance(){
    if(this->game == "connect4"){
        return new games::Connect4();
    } else if (this->game == "breakthrough") {
        return new games::Breakthrough();
    } else {
        throw std::runtime_error("unknown game");
        return nullptr; // fixes linting errors
    }
}

nn::TrainingSample SelfPlay::get_training_sample(
    nn::move_dist normalized_visit_counts,
    std::string moves,
    Board board
){
    auto policy_tensor = this->neural_net->move_map_to_policy_tensor(normalized_visit_counts);
    auto state_tensor = this->neural_net->state_to_tensor(board);    

    nn::TrainingSample ts = {
        policy_tensor,
        state_tensor,
        0,
        board.to_move,
        moves
    };

    return ts;
}

/**
 * @brief Plays a single game
 * 
 * @param thread_idx 
 * @param thread_data 
 */
void SelfPlay::thread_game(int thread_idx, ThreadData * thread_data){
    // decrement games left
    (thread_data->games_left)--;

    game::IGame* game = this->get_game_instance();
    nn::NN * nn_ptr = this->neural_net.get();

    // agent
    eval_f eval_func = make_eval_function(thread_idx, thread_data, nn_ptr);
    Agent * agent = new Agent(game, eval_func);

    int argmax_depth = config::hp["depth_until_pi_argmax"].get<int>();

    std::vector<nn::TrainingSample> samples;

    int num_moves = 0;

    std::stringstream moves;

    while(!game->is_terminal()){

        agent->search(
            config::hp["search_depth"].get<int>()
        );

        auto visit_counts = agent->root_visit_counts();
        auto normalized_visit_counts = utils::softmax_map(visit_counts);

        nn::TrainingSample ts = this->get_training_sample(normalized_visit_counts, moves.str(), game->get_board());
        samples.push_back(ts);

        game::move_id best_move;

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

    this->write_samples(&samples, agent, thread_data, game);

    delete agent;
    delete game;

    std::cout << "Games left: " << thread_data->games_left + thread_data->num_active_threads << std::endl;
}

void SelfPlay::write_samples(
    std::vector<nn::TrainingSample> *training_samples, 
    Agent* agent, 
    ThreadData * thread_data, 
    game::IGame * game
){
    double outcome = agent->outcome_to_value(game->outcome(pp::First));

    for(auto &sample : *training_samples){
        sample.outcome = agent->eval_for_player(outcome, sample.player);
    }

    // now we need to insert the training data into the db
    thread_data->db_mutex.lock();
    this->db->insert_training_samples(training_samples);
    thread_data->db_mutex.unlock();

}


/**
 * @brief Plays games on separate threads until all games are completed
 * 
 * @param thread_idx 
 * @param thread_data
 */
void SelfPlay::thread_play(
    int thread_idx,
    ThreadData * thread_data
) {

    // play games until all games are completed
    while((thread_data->games_left) > 0){
        this->thread_game(thread_idx, thread_data);
    }
    thread_data->q_cv.notify_one();
}
