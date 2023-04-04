#include "./mutual.h"
#include "../utils/utils.h"

void join_threads(std::vector<std::vector<std::thread>*> threads) {
    for (auto t : threads) {
        for (auto& t2 : *t) {
            t2.join();
        }
    }
}

nn::NN* get_neural_net(std::string game, db::DB* db) {

    std::string model_path = utils::string_format(
        "./models/%d.pt",
        db->curr_generation
    );

    std::cout << "making neural net" << std::endl;

    if (game == "connect4") {
        return new nn::Connect4NN(model_path);
    }
    else if (game == "breakthrough") {
        return new nn::BreakthroughNN(model_path);
    }
    else {
        throw std::runtime_error("no neural net exists for this game");
        return nullptr; // for linter 
    }
}

ThreadData* init_thread_data(std::string game_name, int num_games, int generation_num) {
    auto db = new db::DB(generation_num);

    auto nn = get_neural_net(game_name, db);

    return new ThreadData(
        nn,
        db,
        num_games
    );
}

void pop_batch(
    ThreadData* thread_data,
    std::function<int()> bs,
    std::vector<EvalRequest*>* states
) {
    for (int i = 0; i < bs(); i++) {
        auto p = thread_data->eval_q.front();
        thread_data->eval_q.pop();
        states->push_back(p);
    }
}

void dl_thread_work(std::queue<Batch>* batch_queue, ThreadData* thread_data, std::mutex* batch_queue_mutex, std::condition_variable* batch_queue_cv) {
    auto bs = [thread_data]() {
        return (unsigned long)std::min(
            (int)thread_data->num_active_games,
            config::hp["batch_size"].get<int>()
        );
    };

    while (thread_data->num_active_games > 0) {
        std::unique_lock<std::mutex> nn_q_lock(thread_data->q_mutex);

        thread_data->q_cv.wait(nn_q_lock, [&thread_data, &bs]() {
            return thread_data->eval_q.size() >= bs();
            });

        if (bs() == 0) {
            nn_q_lock.unlock();
            break;
        }

        std::vector<EvalRequest*> states;

        pop_batch(thread_data, bs, &states);

        std::vector<at::Tensor> tensors;
        std::vector<std::vector<game::move_id> *> legal_moves;
        for (auto& s : states) {
            tensors.push_back(s->state);
            legal_moves.push_back(s->legal_moves);
        }

        at::Tensor batch = thread_data->neural_net->prepare_batch(tensors);

        batch_queue_mutex->lock();
        batch_queue->push(Batch{ 
            std::move(states), 
            std::move(batch), 
            std::make_pair(at::Tensor(), at::Tensor()),
            std::move(legal_moves)
        });
        // std::cout << "pushed batch to queue" << std::endl;
        batch_queue_mutex->unlock();
        batch_queue_cv->notify_one();
    }
    std::cout << "dl thread done" << std::endl;
}

void nn_thread_work(
    std::queue<Batch>* batch_queue,
    std::queue<Batch>* batch_result_queue,
    ThreadData* thread_data,
    std::mutex* batch_queue_mutex,
    std::condition_variable* batch_queue_cv,
    std::mutex* batch_result_queue_mutex,
    std::condition_variable* batch_result_queue_cv
) {
    while (true) {

        auto cond = [batch_queue]() {
            return batch_queue->size() >= 1;
        };

        std::chrono::milliseconds timeout = std::chrono::milliseconds(0);
        std::unique_lock<std::mutex> lock(*batch_queue_mutex);

        while (!cond()) {
            batch_queue_cv->wait_for(lock, timeout);
            if (thread_data->num_active_games <= 0) {
                lock.unlock();
                std::cout << "nn thread done" << std::endl;
                return;
            }
        }

        Batch batch = batch_queue->front();
        batch_queue->pop();
        lock.unlock();

        if (batch.batch_tensor.is_cuda()) {
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

void eval_batch(Batch* batch, ThreadData* thread_data) {

    auto result = thread_data->neural_net->net_out_to_nnout(
        batch->result.first,
        batch->result.second,
        batch->legal_moves
    );

    if (result.size() != batch->requests.size()) {
        throw std::runtime_error("result size != batch size");
    }

    for (int i = 0; i < (int)batch->requests.size(); i++) {
        batch->requests[i]->result = std::move(result[i]);
        batch->requests[i]->completed = true;
    }

}

void return_thread_work(ThreadData* thread_data, std::queue<Batch>* batch_res_queue, std::mutex* batch_res_queue_mutex, std::condition_variable* batch_res_queue_cv) {


    while (thread_data->num_active_games > 0) {

        auto cond = [batch_res_queue]() {
            return batch_res_queue->size() >= 1;
        };

        std::chrono::milliseconds timeout = std::chrono::milliseconds(0);
        std::unique_lock<std::mutex> lock(*batch_res_queue_mutex);

        while (!cond()) {
            batch_res_queue_cv->wait_for(lock, timeout);
            if (thread_data->num_active_games == 0) {
                lock.unlock();
                std::cout << "return thread donw" << std::endl;
                return;
            }
        }

        std::vector<EvalRequest*> states;

        auto batch = batch_res_queue->front();
        batch_res_queue->pop();

        lock.unlock();

        eval_batch(&batch, thread_data);
    }

}

void start_batching_threads(
    ThreadData* thread_data,
    BatchData* batch_data,
    std::vector<std::thread>& dl_threads,
    std::vector<std::thread>& nn_threads,
    std::vector<std::thread>& return_threads
) {


    int n_dl_threads = config::hp["num_dl_threads"].get<int>();
    for (int i = 0; i < n_dl_threads; i++) {
        dl_threads.push_back(std::thread(dl_thread_work, &batch_data->batch_queue, thread_data, &batch_data->batch_queue_mutex, &batch_data->batch_queue_cv));
    }



    int n_nn_threads = config::hp["num_nn_threads"].get<int>();

    for (int i = 0; i < n_nn_threads; i++) {
        nn_threads.push_back(std::thread(nn_thread_work, &batch_data->batch_queue, &batch_data->batch_result_queue, thread_data, &batch_data->batch_queue_mutex, &batch_data->batch_queue_cv, &batch_data->batch_result_queue_mutex, &batch_data->batch_result_queue_cv));
    }

    int n_return_threads = config::hp["num_return_threads"].get<int>();

    for (int i = 0; i < n_return_threads; i++) {
        return_threads.push_back(std::thread(return_thread_work, thread_data, &batch_data->batch_result_queue, &batch_data->batch_result_queue_mutex, &batch_data->batch_result_queue_cv));
    }
}

game::IGame* get_game_instance(std::string game) {
    if (game == "connect4") {
        return new games::Connect4();
    }
    else if (game == "breakthrough") {
        return new games::Breakthrough();
    }
    else {
        throw std::runtime_error("unknown game");
        return nullptr; // fixes linting errors
    }
}

void queue_request(
    ThreadData* thread_data, 
    Board& board, 
    std::vector<game::move_id> *legal_moves, 
    EvalRequest* request
) {
    request->completed = false;
    request->result = nullptr;
    request->state = thread_data->neural_net->state_to_tensor(board);
    request->legal_moves = legal_moves;

    thread_data->q_mutex.lock();
    thread_data->eval_q.push(request);
    thread_data->q_mutex.unlock();
    thread_data->q_cv.notify_one();
}