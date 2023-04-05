#pragma once

#include <vector>
#include <chrono>
#include <memory>
#include <limits>
#include "./random.h"
#include "./dirichlet.h"
#include "./strings.h"
#include "./thread_queue.h"
#include "./timer.h"
#include "./thread_pool.h"


namespace utils {
    /*
        Check if dir exists.
    */
    bool dir_exists(const std::string& path);    

    /*
     * Timing.
     */
    template<typename F, typename... Args>
    [[maybe_unused]] auto time_function(F func, Args&&... args){
        using namespace std::chrono;
        high_resolution_clock::time_point time_start = high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        high_resolution_clock::time_point time_end = high_resolution_clock::now();
        return duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
    }

    // class Timer {
    // public:
    //     void start() { stop_ = start_ = std::chrono::high_resolution_clock::now(); }
    //     void stop() { stop_ = std::chrono::high_resolution_clock::now(); }
    //     [[nodiscard]] uint64_t duration_ms() const {
    //         return std::chrono::duration_cast<std::chrono::milliseconds>(stop_ - start_).count();
    //     }
    // private:
    //     std::chrono::high_resolution_clock::time_point start_;
    //     std::chrono::high_resolution_clock::time_point stop_;
    // };

    template <typename T, typename Func>
    int arg_max(const std::vector<T>& vec, Func f) {
        return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end(),
                                                                       [&f](const T& a, const T& b){
                                                                           return f(a) < f(b);
                                                                       })));
    }

    template <typename T, typename Func>
    int arg_min(const std::vector<T>& vec, Func f) {
        return static_cast<int>(std::distance(vec.begin(), min_element(vec.begin(), vec.end(),
                                                                       [&f](const T& a, const T& b){
                                                                           return f(a) < f(b);
                                                                       })));
    }
}




