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
#include "../global.h"
#include <stdexcept>
#include <thread>
#include <cstring>
#include <unistd.h>
#include <memory>

struct EvalRequest {
    bool completed;
    at::Tensor state;
    std::unique_ptr<nn::NNOut> result;
};

struct Batch {
    std::vector<EvalRequest*> requests;
    at::Tensor batch_tensor;
    std::pair<at::Tensor, at::Tensor> result;
};

struct BatchData {
    std::queue<Batch> batch_queue;
    std::mutex batch_queue_mutex;
    std::condition_variable batch_queue_cv;
    std::queue<Batch> batch_result_queue;
    std::mutex batch_result_queue_mutex;
    std::condition_variable batch_result_queue_cv;
};


struct ThreadData {
    std::queue<EvalRequest*> eval_q;
    std::mutex q_mutex;
    std::mutex db_mutex;
    std::mutex start_game_mutex;
    std::condition_variable q_cv;
    std::atomic<int> games_left;
    std::atomic<int> num_active_games;
    db::DB * db;
    nn::NN * neural_net;

    ThreadData(
        int num_threads, 
        nn::NN * neural_net,
        db::DB * db,
        int num_games
    ): 
        q_mutex(), 
        db_mutex(), 
        start_game_mutex(), 
        q_cv(), 
        db(db), 
        neural_net(neural_net) 
    {
        // set all variables
        this->games_left = num_games;
        this->num_active_games = config::hp["self_play_num_threads"].get<int>() * config::hp["games_per_thread"].get<int>();
        this->eval_q = std::queue<EvalRequest*>();
    }

    ~ThreadData(){
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
    ThreadData *thread_data,
    std::string game_name
);

void join_threads(std::vector<std::vector<std::thread>*> threads);

void self_play_start_threads(
    std::vector<std::thread> &threads, 
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
    std::vector<EvalRequest*> *states
);

void start_batching_threads(
    ThreadData * thread_data, 
    BatchData * batch_data,
    std::vector<std::thread> &dl_threads, 
    std::vector<std::thread> &nn_threads,
    std::vector<std::thread> &return_threads
);

void eval_batch(Batch * batch, ThreadData *thread_data);

void self_play_active_thread_work(ThreadData *thread_data, std::queue<Batch> *batch_queue, std::mutex *batch_queue_mutex, std::condition_variable *batch_queue_cv);

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

struct ThreadEvalData{
    Agent * agent = nullptr;
    game::IGame * game = nullptr;
    EvalRequest request;
    GroundTruthRequest gt_request;
    bool dead_game;
    nn::NNOut first_nn_out;
    bool first_nn_out_set;

    ThreadEvalData(){
        dead_game = false;
        first_nn_out_set = false;
    }
};

void queue_request(ThreadData * thread_data, Board &board, EvalRequest *request){
    request->completed = false;
    request->result = nullptr;
    request->state = thread_data->neural_net->state_to_tensor(board);

    thread_data->q_mutex.lock();
    thread_data->eval_q.push(request);
    thread_data->q_mutex.unlock();
    thread_data->q_cv.notify_one();   
}


void init_eval_games(ThreadEvalData * data, ThreadData * thread_data, EvalData * eval_data, int num_games){
     // start the games
    for(int i = 0; i < num_games; i++){
        auto &ed = data[i];

        if(DEBUG){
            std::cout << "Starting game " << i << std::endl;
        }

        eval_data->board_queue_mutex.lock();

        if(eval_data->board_queue.empty()){
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

        for(char mv: req.moves){
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
    ThreadData * thread_data,
    EvalData * eval_data
) {
    if(DEBUG){
        std::cout << "Started eval thread" << std::endl;
    }

    const int num_games = config::hp["games_per_thread"].get<int>();

    ThreadEvalData data[num_games];

    init_eval_games(data, thread_data, eval_data, num_games);

    if(DEBUG){
        std::cout << "Started games" << std::endl;
    }

    while(true){
        bool all_dead = true;
        for(int i = 0; i < num_games; i++){
            auto &ed = data[i];

            if(!ed.dead_game){
                all_dead = false;
            }

            if(ed.dead_game || !ed.request.completed){
                usleep(0); // give other threads a chance to run
                continue;
            }

            // get answer
            if(DEBUG){
                std::cout << "Got answer" << std::endl;
            }

            std::unique_ptr<nn::NNOut> answer = std::move(ed.request.result);

            if(!ed.first_nn_out_set){
                ed.first_nn_out = *answer;
                ed.first_nn_out_set = true;
            }

            auto [done, board] = ed.agent->step(std::move(answer));

            if(done){
                if(DEBUG){
                    std::cout << "Evaluated board" <<  std::endl;
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

                if(eval_data->board_queue.empty()){
                    eval_data->board_queue_mutex.unlock();
                    ed.dead_game = true;
                    thread_data->num_active_games--;
                    thread_data->q_cv.notify_all();
                    continue;

                } else {
                    // start new game
                    auto req = eval_data->board_queue.front();
                    eval_data->board_queue.pop();
                    eval_data->board_queue_mutex.unlock();

                    ed.gt_request = req;
                    ed.first_nn_out_set = false;

                    ed.game = get_game_instance("connect4");

                    for(char mv: req.moves){
                        ed.game->make(mv - '0');
                    }

                    ed.agent = new Agent(ed.game, false);

                    std::tie(done, board) = ed.agent->init_mcts(config::hp["eval_search_depth"].get<int>());
                }
            }
            queue_request(thread_data, board, &ed.request);
        }
        if(all_dead){
            break;
        }
    }

    std::cout << "Eval thread done" << std::endl;
}



void start_eval_threads(
    ThreadData * thread_data, 
    EvalData * eval_data, 
    std::vector<std::thread> &threads, 
    int num_threads
){
    for(int i = 0; i < num_threads; i++){
        threads.push_back(std::thread(thread_eval, thread_data, eval_data));
    }
}

void sim::eval_targets(std::string eval_targets_filename, int generation_num){

    int num_threads = config::hp["self_play_num_threads"].get<int>();
    int num_games = config::hp["self_play_num_games"].get<int>();

    EvalData eval_data;

    auto eval_data_queue = make_eval_data_queue(eval_targets_filename);
    eval_data.board_queue = eval_data_queue;

    std::vector<std::thread> threads;

    auto thread_data = init_thread_data("connect4", num_threads, num_games, generation_num);

    std::vector<std::thread> dl_threads;
    std::vector<std::thread> nn_threads;
    std::vector<std::thread> return_threads;

    BatchData batch_data = BatchData();

    if(DEBUG){
        std::cout << "Starting batching threads" << std::endl;
    }

    start_batching_threads(thread_data, &batch_data, dl_threads, nn_threads, return_threads);

    // self_play_start_threads(threads, thread_data, num_threads, game);

    if(DEBUG){
        std::cout << "Starting eval threads" << std::endl;
    }

    start_eval_threads(thread_data, &eval_data, threads, num_threads);

    join_threads({&dl_threads, &nn_threads, &return_threads, &threads});

}


void dl_thread_work(std::queue<Batch> * batch_queue, ThreadData * thread_data, std::mutex * batch_queue_mutex, std::condition_variable * batch_queue_cv){
    auto bs = [thread_data](){
        return (unsigned long)std::min(
            (int)thread_data->num_active_games, 
            config::hp["batch_size"].get<int>()
        );
    };

    while(thread_data->num_active_games > 0){
        std::unique_lock<std::mutex> nn_q_lock(thread_data->q_mutex);

        thread_data->q_cv.wait(nn_q_lock, [&thread_data, &bs](){
            return thread_data->eval_q.size() >= bs();
        });

        if(bs() == 0){
            nn_q_lock.unlock();
            break;
        }

        std::vector<EvalRequest *> states;

        pop_batch(thread_data, bs, &states);

        std::vector<at::Tensor> tensors;
        for(auto &s: states){
            tensors.push_back(s->state);
        }

        at::Tensor batch = thread_data->neural_net->prepare_batch(tensors);

        batch_queue_mutex->lock();
        batch_queue->push(Batch{states, batch, std::make_pair(at::Tensor(), at::Tensor())});
        // std::cout << "pushed batch to queue" << std::endl;
        batch_queue_mutex->unlock();
        batch_queue_cv->notify_one();
    }
    std::cout << "dl thread done" << std::endl;
}

void nn_thread_work(
    std::queue<Batch> * batch_queue, 
    std::queue<Batch> * batch_result_queue, 
    ThreadData * thread_data, 
    std::mutex * batch_queue_mutex, 
    std::condition_variable * batch_queue_cv, 
    std::mutex * batch_result_queue_mutex, 
    std::condition_variable * batch_result_queue_cv
){
    while(true){

        auto cond = [batch_queue](){
            return batch_queue->size() >= 1;
        };

        std::chrono::milliseconds timeout = std::chrono::milliseconds(0);
        std::unique_lock<std::mutex> lock(*batch_queue_mutex);

        while (!cond()) {
            batch_queue_cv->wait_for(lock, timeout);
            if(thread_data->num_active_games <= 0){
                lock.unlock();
                std::cout << "nn thread done" << std::endl;
                return;
            }
        }

        Batch batch = batch_queue->front();
        batch_queue->pop();
        lock.unlock();

        if(batch.batch_tensor.is_cuda()){
            throw std::runtime_error("batch tensor is on gpu");
        }

        auto outputs = thread_data->neural_net->run_batch(batch.batch_tensor);

        // try to clear memory
        batch.batch_tensor.cpu();
        batch.batch_tensor.reset();
        batch.batch_tensor = at::Tensor();
        
        at::Tensor t1 = outputs.at(0).toTensor();
        at::Tensor t2 = outputs.at(1).toTensor();

        batch.result = std::make_pair(
            t1.cpu(),
            t2.cpu()
        );

        t1.reset();
        t2.reset();

        batch_result_queue_mutex->lock();
        batch_result_queue->push(batch);
        batch_result_queue_mutex->unlock();
        batch_result_queue_cv->notify_one();
    }

}

void start_batching_threads(
    ThreadData * thread_data, 
    BatchData * batch_data,
    std::vector<std::thread> &dl_threads, 
    std::vector<std::thread> &nn_threads,
    std::vector<std::thread> &return_threads
){


    int n_dl_threads = config::hp["num_dl_threads"].get<int>();
    for(int i = 0; i < n_dl_threads; i++){
        dl_threads.push_back(std::thread(dl_thread_work, &batch_data->batch_queue, thread_data, &batch_data->batch_queue_mutex, &batch_data->batch_queue_cv));
    }



    int n_nn_threads = config::hp["num_nn_threads"].get<int>();

    for(int i = 0; i < n_nn_threads; i++){
        nn_threads.push_back(std::thread(nn_thread_work, &batch_data->batch_queue, &batch_data->batch_result_queue, thread_data, &batch_data->batch_queue_mutex, &batch_data->batch_queue_cv, &batch_data->batch_result_queue_mutex, &batch_data->batch_result_queue_cv));
    }

    int n_return_threads = config::hp["num_return_threads"].get<int>();

    for(int i = 0; i < n_return_threads; i++){
        return_threads.push_back(std::thread(self_play_active_thread_work, thread_data, &batch_data->batch_result_queue, &batch_data->batch_result_queue_mutex, &batch_data->batch_result_queue_cv));
    }
}

void join_threads(std::vector<std::vector<std::thread>*> threads){
    for(auto t: threads){
        for(auto &t2: *t){
            t2.join();
        }
    }
}

/**
 * @brief Starts the self play process
 * 
 * @param game
 */
void sim::self_play(std::string game){

    int num_threads = config::hp["self_play_num_threads"].get<int>();
    int num_games = config::hp["self_play_num_games"].get<int>();

    std::vector<std::thread> threads;

    auto thread_data = init_thread_data(game, num_threads, num_games);

    std::vector<std::thread> dl_threads;
    std::vector<std::thread> nn_threads;
    std::vector<std::thread> return_threads;

    BatchData batch_data = BatchData();

    start_batching_threads(thread_data, &batch_data, dl_threads, nn_threads, return_threads);

    self_play_start_threads(threads, thread_data, num_threads, game);

    join_threads({&dl_threads, &nn_threads, &return_threads, &threads});
}



ThreadData * init_thread_data(std::string game_name, int num_threads, int num_games, int generation_num){
    auto db = new db::DB(game_name, generation_num);

    auto nn = get_neural_net(game_name, db);


    return new ThreadData(
        num_threads, 
        nn,
        db,
        num_games
    );
}


nn::NN * get_neural_net(std::string game, db::DB *db){

    std::string model_path = utils::string_format(
        "./models/%d.pt", 
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
void self_play_start_threads(std::vector<std::thread> &threads, ThreadData *thread_data, int num_threads, std::string game_name)
{
    for(int i = 0; i < num_threads; i++){
        threads.push_back(std::thread(
            &thread_play, thread_data, game_name
        ));
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
    std::vector<EvalRequest*> *states
){
    for(int i = 0; i < bs(); i++){
        auto p = thread_data->eval_q.front();
        thread_data->eval_q.pop();
        states->push_back(p);
    }
}

/**
 * @brief Evaluate a batch of states.
 * 
 * @param states 
 * @param thread_data 
 */
void eval_batch(Batch * batch, ThreadData *thread_data)
{


    auto result = thread_data->neural_net->net_out_to_nnout(
        batch->result.first,
        batch->result.second
    );

    if(result.size() != batch->requests.size()){
        throw std::runtime_error("result size != batch size");
    }

    for(int i = 0; i < (int)batch->requests.size(); i++){
        batch->requests[i]->result = std::move(result[i]);
        batch->requests[i]->completed = true;
    }
}



/**
 * @brief Main thread work loop. Waits until the queue contains a batch, pops it and evaluates it.
 * 
 * @param thread_data 
 * @param bs 
 */
void self_play_active_thread_work(ThreadData *thread_data, std::queue<Batch> *batch_res_queue, std::mutex *batch_res_queue_mutex, std::condition_variable *batch_res_queue_cv) {


    while(thread_data->num_active_games > 0){

        auto cond = [batch_res_queue](){
            return batch_res_queue->size() >= 1;
        };

        std::chrono::milliseconds timeout = std::chrono::milliseconds(0);
        std::unique_lock<std::mutex> lock(*batch_res_queue_mutex);

        while (!cond()) {
            batch_res_queue_cv->wait_for(lock, timeout);
            if(thread_data->num_active_games == 0){
                lock.unlock();
                std::cout << "return thread donw" << std::endl;
                return;
            }
        }

        std::vector<EvalRequest *> states;

        auto batch = batch_res_queue->front();
        batch_res_queue->pop();

        lock.unlock();

        eval_batch(&batch, thread_data);
    }

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
    ThreadData * thread_data,
    std::string game_name
) {

    const int num_games = config::hp["games_per_thread"].get<int>();

    Agent * agents[num_games];
    game::IGame * games[num_games];
    EvalRequest requests[num_games];
    bool dead_game[num_games];
    std::vector<db::TrainingSample> samples[num_games];
    std::stringstream moves[num_games];
    int num_moves[num_games];
    memset(num_moves, 0, sizeof(num_moves));

    // start the games
    for(int i = 0; i < num_games; i++){
        thread_data->start_game_mutex.lock();
        if(thread_data->games_left <= 0){
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
        auto [done, board] = agents[i]->init_mcts(config::hp["search_depth"].get<int>());

        queue_request(thread_data, board, &requests[i]);
    }

    while(true){
        bool all_dead = true;
        for(int i = 0; i < num_games; i++){

            if(!dead_game[i]){
                all_dead = false;
            }

            if(dead_game[i] || !requests[i].completed){
                usleep(0); // give other threads a chance to run
                continue;
            }



            // get answer
            std::unique_ptr<nn::NNOut> answer = std::move(requests[i].result);
            auto [done, board] = agents[i]->step(std::move(answer));

            while(done){
                // agent is ready to move
                auto visit_counts = agents[i]->root_visit_counts();
                auto normalized_visit_counts = utils::softmax_map(visit_counts);

                samples[i].push_back(get_training_sample(
                    normalized_visit_counts, 
                    moves[i].str(), 
                    games[i]->get_board(),
                    thread_data->neural_net
                ));

                auto best_move = select_move(normalized_visit_counts, num_moves[i]);

                moves[i] << games[i]->move_as_str(best_move) << ";";

                games[i]->make(best_move);
                agents[i]->update_tree(best_move);

                num_moves[i]++;
                // std::cout << "Made move " << num_moves[i] << std::endl;

                bool game_completed = games[i]->is_terminal();

                if(game_completed){
                    write_samples(&samples[i], agents[i], thread_data, games[i], num_moves[i]);

                    // restart game
                    samples[i].clear();
                    num_moves[i] = 0;
                    delete agents[i];
                    delete games[i];
                    std::cout << "Games left: " << thread_data->games_left + thread_data->num_active_games << std::endl;


                    thread_data->start_game_mutex.lock();
                    if(thread_data->games_left <= 0){
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
                }

                auto res = agents[i]->init_mcts(config::hp["search_depth"].get<int>());
                done = res.first;
                board = res.second;
            }

            queue_request(thread_data, board, &requests[i]);

            cnt:;
        }
        if(all_dead){
            break;
        }
    }

    std::cout << "Game thread done" << std::endl;
}
