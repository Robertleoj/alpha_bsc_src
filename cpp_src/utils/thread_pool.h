#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <functional>

namespace utils {

    template<typename T, typename R>
    class ThreadPool {
    public:
        ThreadPool(size_t numThreads) {

            for (size_t i = 0; i < numThreads; ++i) {
                threads.emplace_back([this]() {
                    std::mutex m;
                    std::unique_lock<std::mutex> lock(m);
                    start_cond.wait(lock, [this]() { return !tasks.empty(); });
                    lock.unlock();

                    while (true) {
                        std::function<R()> task;
                        int item_idx;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            if (tasks.empty() || done) {
                                return;
                            }
                            std::tie(item_idx, task) = tasks.front();
                            tasks.pop();
                        }

                        auto res = std::move(task());
                        result_mutex.lock();
                        results[item_idx] = std::move(res);
                        result_mutex.unlock();

                        tasks_done++;
                        if (tasks_done == expected_results) {
                            done = true;
                            result_condition.notify_one();
                        }
                    }
                });
            }
        }

        ~ThreadPool() {
            for (std::thread& thread : threads) {
                thread.join();
            }
        }

        std::vector<R> map(const std::vector<T>& inputs, std::function<R(T)> func) {
            tasks_done = 0;
            done = false;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                expected_results = inputs.size();
                results = std::vector<R>(expected_results);
                tasks = std::queue<std::pair<int, std::function<R()>>>();
                int i = 0;
                for (const T& input : inputs) {
                    tasks.push(std::make_pair(i, [input, func](){
                        return std::move(func(input));
                    }));
                    i++;
                }
            }
            start_cond.notify_all();
            {
                auto timeout = std::chrono::milliseconds(5);
                std::unique_lock<std::mutex> lock(result_mutex);
                while(!done){
                    result_condition.wait_for(lock, timeout);
                }
                // result_condition.wait(lock, [this]() { return (bool)done; });
            }
            return std::move(results);
        }

    private:
        size_t expected_results;
        std::vector<R> results;
        std::queue<std::pair<int, std::function<R()>>> tasks;
        std::vector<std::thread> threads;
        std::mutex queue_mutex;
        std::mutex result_mutex;
        std::atomic<int> tasks_done;
        std::atomic<bool> done;
        std::condition_variable start_cond;
        std::condition_variable result_condition;
    };
}