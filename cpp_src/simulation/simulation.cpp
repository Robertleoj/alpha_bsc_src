#include "../NN/nn.h"
#include "../NN/connect4_nn.h"
#include "./simulation.h"
#include "../utils/utils.h"
#include "../games/game.h"
#include "../games/connect4.h"
#include "../games/breakthrough.h"
#include "../config/config.h"
#include "../base/types.h"
#include "../utils/utils.h"
#include <stdexcept>
#include <thread>
#include <cstring>

typedef std::pair<int, at::Tensor> eval_request;

struct ThreadData {
    std::queue<eval_request> eval_q;
    std::mutex q_mutex;
    std::mutex db_mutex;
    bool *req_completed;
    std::mutex req_completed_mutex;
    std::unique_ptr<nn::NNOut> *evaluations;
    std::condition_variable eval_cv;
    std::condition_variable q_cv;
    std::mutex results_mutex;
    std::atomic<int> games_left;
    std::atomic<int> num_active_threads;
    db::DB * db;
    nn::NN * neural_net;

    ThreadData(
        int num_threads, 
        nn::NN * neural_net,
        db::DB * db,
        int num_games, 
        bool *req_completed,
        std::unique_ptr<nn::NNOut> *evaluations
    )
        : q_mutex(), db_mutex(), req_completed(req_completed),
            req_completed_mutex(), evaluations(evaluations), eval_cv(), q_cv(),
            results_mutex(), db(db), neural_net(neural_net) {
        // set all variables
        this->games_left = num_games;
        this->num_active_threads = num_threads;
        this->eval_q = std::queue<eval_request>();

    }

    ~ThreadData(){
        delete[] req_completed;
        delete[] evaluations;
    }
};

game::IGame *get_game_instance(std::string game);

struct GroundTruthRequest {
    std::string moves;
    std::vector<double> ground_truth;
    double value;
};

struct EvalData {
    std::queue<GroundTruthRequest> board_queue;
    std::mutex board_queue_mutex;
};


// declarations
void thread_play(
    int thread_idx, 
    ThreadData *thread_data,
    std::string game_name
);

eval_f make_eval_function(
    int thread_idx, 
    ThreadData *thread_data
);

void self_play_start_threads(
    std::thread *threads, 
    ThreadData *thread_data, 
    int num_threads,
    std::string game_name
);

void thread_game(
    int thread_idx, 
    ThreadData *thread_data,
    std::string game
);


void write_samples(
    std::vector<db::TrainingSample> *, 
    Agent *, 
    ThreadData *,
    game::IGame *,
    int num_moves
);

db::TrainingSample get_training_sample(
    nn::move_dist normalized_visit_counts,
    std::string moves,
    Board board,
    nn::NN * neural_net
);

game::move_id select_move(
    nn::move_dist, 
    int num_moves
);

void pop_batch(
    ThreadData *thread_data, 
    std::function<int()> bs,
    std::vector<int> *thread_indices,
    std::vector<at::Tensor> *states
);

void eval_batch(
    std::vector<at::Tensor> &states,
    std::vector<int> &thread_indices, 
    ThreadData *thread_data
);

void self_play_active_thread_work(ThreadData *thread_data);

nn::NN * get_neural_net(
    std::string game, 
    db::DB *db
);

ThreadData * init_thread_data(std::string game_name, int num_threads, int num_games, int generation_num = -1);



// implementations


/**
 * @brief Load data from json file and make a queue of ground truth requests
 * 
 * @param ground_truth_filename 
 * @return std::queue<GroundTruthRequest> 
 */
std::queue<GroundTruthRequest> make_eval_data_queue(std::string ground_truth_filename){
    // read in the ground truth file - it is a json file
    std::ifstream ground_truth_file(ground_truth_filename);
    std::cout << "Reading ground truth file: " << ground_truth_filename << std::flush;

    nlohmann::json ground_truth_json = nlohmann::json::parse(ground_truth_file);

    std::cout << colors::GREEN << " [DONE]" << colors::RESET << std::endl;

    std::queue<GroundTruthRequest> eval_data_queue;

    for(auto &game: ground_truth_json.get<std::vector<nlohmann::json>>()){
        std::string moves = game["moves"].get<std::string>();
        std::vector<double> ground_truth = game["target"].get<std::vector<double>>();
        double value = game["value"].get<double>();    
        eval_data_queue.push(GroundTruthRequest{moves, ground_truth, value});
    }
    return eval_data_queue;
}

std::vector<double> get_policy_vector(nn::move_dist prior_map){
    std::vector<double> out;
    for(game::move_id move = 1; move <= 7; move++){
        out.push_back(prior_map[move]);
    }

    return out;
}

double get_ce_loss(std::vector<double> vec1, std::vector<double> vec2){
    auto cross_entropy = torch::nn::CrossEntropyLoss();
    return cross_entropy(
        torch::tensor(vec1), 
        torch::tensor(vec2)
    ).item<double>();
}


void write_evaluation_to_db(
    ThreadData * thread_data,
    GroundTruthRequest * gt,
    std::vector<double> &policy_prior,
    double nn_value,
    std::vector<double> &policy_mcts,
    double mcts_value
){

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


void evaluate_request(GroundTruthRequest * gt, ThreadData * thread_data, int thread_idx){

    game::IGame* game = get_game_instance("connect4");

    for(char mv: gt->moves){
        game->make(mv - '0');
    }

    eval_f eval_func = make_eval_function(thread_idx, thread_data);

    Agent * agent = new Agent(game, eval_func, false);

    agent->search(config::hp["eval_search_depth"].get<int>());

    auto visit_counts = agent->root_visit_counts();
    auto pol_mcts = utils::softmax_map(visit_counts);

    std::vector<double> pol_mcts_vec = get_policy_vector(pol_mcts); // THIS

    auto eval_out = eval_func(game->get_board());

    auto pol_prior = eval_out->p;
    std::vector<double> pol_prior_vec = get_policy_vector(pol_prior); // THIS

    double nn_value = eval_out->v; // THIS

    double mcts_value = agent->tree->root->value_approx; // THIS

    write_evaluation_to_db(
        thread_data,
        gt,
        pol_prior_vec, 
        nn_value,
        pol_mcts_vec, 
        mcts_value
    );
    

    delete agent;
    delete game;
}

void thread_eval(
    int thread_idx,
    ThreadData * thread_data,
    EvalData * eval_data
) {


    while(true){
        eval_data->board_queue_mutex.lock();
        if(eval_data->board_queue.empty()){
            eval_data->board_queue_mutex.unlock();
            break;
        }

        GroundTruthRequest req = eval_data->board_queue.front();

        eval_data->board_queue.pop();
        eval_data->board_queue_mutex.unlock();


        evaluate_request(&req, thread_data, thread_idx);
    }

    thread_data->num_active_threads--;
    thread_data->q_cv.notify_one();
}



void start_eval_threads(
    ThreadData * thread_data, 
    EvalData * eval_data, 
    std::thread * threads, 
    int num_threads
){
    for(int i = 0; i < num_threads; i++){
        threads[i] = std::thread(thread_eval, i, thread_data, eval_data);
    }
}

void sim::eval_targets(std::string eval_targets_filename, int generation_num){
    int num_threads = config::hp["num_parallel_games"].get<int>();
    int num_games = config::hp["self_play_num_games"].get<int>();

    auto eval_data_queue = make_eval_data_queue(eval_targets_filename);

    EvalData eval_data;
    eval_data.board_queue = eval_data_queue;

    std::thread threads[num_threads];
    auto thread_data = init_thread_data("connect4", num_threads, num_games, generation_num);

    start_eval_threads(thread_data, &eval_data, threads, num_threads);

    self_play_active_thread_work(thread_data);
    std::cout << std::endl;

    for(auto &t: threads){
        t.join();
    }

    delete thread_data;
}


/**
 * @brief Starts the self play process
 * 
 * @param game
 */
void sim::self_play(std::string game){

    int num_threads = config::hp["num_parallel_games"].get<int>();
    int num_games = config::hp["self_play_num_games"].get<int>();

    std::thread threads[num_threads];

    auto thread_data = init_thread_data(game, num_threads, num_games);

    self_play_start_threads(threads, thread_data, num_threads, game);

    self_play_active_thread_work(thread_data);

    for(auto &t: threads){
        t.join();
    }
}

ThreadData * init_thread_data(std::string game_name, int num_threads, int num_games, int generation_num){
    auto db = new db::DB(game_name, generation_num);

    auto nn = get_neural_net(game_name, db);

    bool * req_completed = new bool[num_threads];
    memset(req_completed, false, num_threads * sizeof(bool));

    auto evaluations = new std::unique_ptr<nn::NNOut>[num_threads];

    return new ThreadData(
        num_threads, 
        nn,
        db,
        num_games, 
        req_completed, 
        evaluations
    );
}


nn::NN * get_neural_net(std::string game, db::DB *db){

    std::string model_path = utils::string_format(
        "../models/%s/%d.pt", 
        game.c_str(),
        db->curr_generation
    );

    std::cout << "making neural net" << std::endl;

    if(game == "connect4"){
        return new nn::Connect4NN(model_path);
    } else {
        throw std::runtime_error("no neural net exists for this game");
        return nullptr; // for linter 
    }
}

/**
 * @brief Starts `num_theads` threads, each of which plays a game of self-play.
 * 
 * @param threads 
 * @param thread_data 
 * @param num_threads 
 */
void self_play_start_threads(std::thread *threads, ThreadData *thread_data, int num_threads, std::string game_name)
{
    for(int i = 0; i < num_threads; i++){
        threads[i] = std::thread(
            &thread_play, i, thread_data, game_name
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
void pop_batch(
    ThreadData *thread_data, 
    std::function<int()> bs, 
    std::vector<int> *thread_indices, 
    std::vector<at::Tensor> *states
    )
{
    for(int i = 0; i < bs(); i++){
        auto p = thread_data->eval_q.front();
        thread_data->eval_q.pop();

        thread_indices->push_back(p.first);
        states->push_back(p.second);
    }
}

/**
 * @brief Evaluate a batch of states.
 * 
 * @param states 
 * @param thread_indices 
 * @param thread_data 
 */
void eval_batch(std::vector<at::Tensor> &states, std::vector<int> &thread_indices, ThreadData *thread_data)
{
    auto result = thread_data->neural_net->eval_tensors(states);
        for(int i = 0; i < (int)thread_indices.size(); i++){
            int thread_idx = thread_indices[i];

            thread_data->evaluations[thread_idx] = std::move(result[i]);

            thread_data->req_completed[thread_idx] = true;
        }
        thread_data->eval_cv.notify_all();
}

/**
 * @brief Main thread work loop. Waits until the queue contains a batch, pops it and evaluates it.
 * 
 * @param thread_data 
 * @param bs 
 */
void self_play_active_thread_work(ThreadData *thread_data) {

    auto bs = [thread_data](){
        return (unsigned long)std::min(
            (int)thread_data->num_active_threads, 
            config::hp["batch_size"].get<int>()
        );
    };

    while(thread_data->num_active_threads > 0){
        std::unique_lock<std::mutex> nn_q_lock(thread_data->q_mutex);

        thread_data->q_cv.wait(nn_q_lock, [&thread_data, &bs](){
            return thread_data->eval_q.size() >= bs();
        });

        if(bs() == 0){
            nn_q_lock.unlock();
            break;
        }

        std::vector<int> thread_indices;
        std::vector<at::Tensor> states;

        pop_batch(thread_data, bs, &thread_indices, &states);

        nn_q_lock.unlock();

        eval_batch(states, thread_indices, thread_data);
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
eval_f make_eval_function(int thread_idx, ThreadData * thread_data){
    nn::NN * nn_ptr = thread_data->neural_net;
    return [
        thread_idx, 
        thread_data,
        nn_ptr
    ](Board b){
        
        // Make thread create the tensor from state so main thread does not have to
        auto t = nn_ptr->state_to_tensor(b).cuda();

        // Put item in queue
        thread_data->q_mutex.lock();
        thread_data->eval_q.push({thread_idx, t});
        thread_data->q_mutex.unlock();

        // notify main thread that there is an item in the queue
        thread_data->q_cv.notify_one();

        // Wait for result
        std::mutex m;
        std::unique_lock<std::mutex> lq(m);

        // Wait for main thread to notify that the result is ready
        thread_data->eval_cv.wait(lq, [thread_data, thread_idx](){
            return thread_data->req_completed[thread_idx];
        });
        
        // Reset flag
        thread_data->req_completed[thread_idx] = false;
        lq.unlock();

        // Get results
        // thread_data->results_mutex.lock();
        auto result = std::move(thread_data->evaluations[thread_idx]);
        thread_data->evaluations[thread_idx] = nullptr;
        // thread_data->results_mutex.unlock();

        return result;
    };   
}

/**
 * @brief Gets the correct game instance based on the game name (this->game).
 * 
 * @return game::IGame* 
 */
game::IGame * get_game_instance(std::string game){
    if(game == "connect4"){
        return new games::Connect4();
    } else if (game == "breakthrough") {
        return new games::Breakthrough();
    } else {
        throw std::runtime_error("unknown game");
        return nullptr; // fixes linting errors
    }
}

/**
 * @brief Creates a training sample from the given data.
 * 
 * @param normalized_visit_counts 
 * @param moves 
 * @param board 
 * @return nn::TrainingSample 
 */
db::TrainingSample get_training_sample(
    nn::move_dist normalized_visit_counts,
    std::string moves,
    Board board,
    nn::NN * neural_net
){
    auto policy_tensor = neural_net->move_map_to_policy_tensor(normalized_visit_counts);
    auto state_tensor = neural_net->state_to_tensor(board);    

    db::TrainingSample ts = {
        policy_tensor,
        state_tensor,
        0,
        board.to_move,
        moves
    };

    return ts;
}

/**
 * @brief Selects a move based on the visit counts and the current number of moves.
 * 
 * @param visit_count_dist 
 * @param num_moves 
 * @return game::move_id 
 */
 game::move_id select_move(nn::move_dist visit_count_dist, int num_moves) {

    int argmax_depth = config::hp["depth_until_pi_argmax"].get<int>();

    if(num_moves < argmax_depth){
        return utils::sample_multinomial(visit_count_dist);
    } else {
        return utils::argmax_map(visit_count_dist);
    }
}


/**
 * @brief Plays a single game
 * 
 * @param thread_idx 
 * @param thread_data 
 */
void thread_game(int thread_idx, ThreadData * thread_data, std::string game_name){
    // decrement games left
    (thread_data->games_left)--;

    game::IGame* game = get_game_instance(game_name);

    eval_f eval_func = make_eval_function(thread_idx, thread_data);
    Agent * agent = new Agent(game, eval_func);

    std::vector<db::TrainingSample> samples;

    std::stringstream moves;
    int num_moves = 0;
    while(!game->is_terminal()){

        agent->search(config::hp["search_depth"].get<int>());

        auto visit_counts = agent->root_visit_counts();
        auto normalized_visit_counts = utils::softmax_map(visit_counts);

        samples.push_back(get_training_sample(
            normalized_visit_counts, 
            moves.str(), 
            game->get_board(),
            thread_data->neural_net
        ));

        auto best_move = select_move(normalized_visit_counts, num_moves);

        moves << game->move_as_str(best_move) << ";";

        game->make(best_move);
        agent->update_tree(best_move);

        num_moves++;
    }

    write_samples(&samples, agent, thread_data, game, num_moves);

    delete agent;
    delete game;

}

/**
 * @brief Writes the training samples to the database
 * 
 * @param training_samples 
 * @param agent 
 * @param thread_data 
 * @param game 
 */
void write_samples(
    std::vector<db::TrainingSample> *training_samples, 
    Agent* agent, 
    ThreadData * thread_data, 
    game::IGame * game,
    int num_moves
){
    double outcome = agent->outcome_to_value(game->outcome(pp::First));

    int i = 0;
    for(auto &sample : *training_samples){
        sample.outcome = agent->eval_for_player(outcome, sample.player);
        sample.moves_left = num_moves - i; 
        i++;
    }

    // now we need to insert the training data into the db
    thread_data->db_mutex.lock();
    thread_data->db->insert_training_samples(training_samples);
    thread_data->db_mutex.unlock();
}


/**
 * @brief Plays games on separate threads until all games are completed
 * 
 * @param thread_idx 
 * @param thread_data
 */
void thread_play(
    int thread_idx,
    ThreadData * thread_data,
    std::string game_name
) {

    while((thread_data->games_left) > 0){
        thread_game(thread_idx, thread_data, game_name);
        std::cout << "finished game" << std::endl;
        std::cout << "Games left: " 
                  << thread_data->games_left + thread_data->num_active_threads 
                  << std::endl;
    }
    thread_data->num_active_threads--;
    thread_data->q_cv.notify_one();
}
