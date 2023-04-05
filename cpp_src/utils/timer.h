#pragma once
#include <iostream>
#include <chrono>

namespace utils {
    class Timer {
    public:
        Timer() {}

        void start() {
            start_timepoint = std::chrono::high_resolution_clock::now();
        }

        void stop() {
            end_timepoint = std::chrono::high_resolution_clock::now();
        }

        void print() {
            auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_timepoint).time_since_epoch().count();
            auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_timepoint).time_since_epoch().count();
            auto duration = end - start;
            double ms = duration * 0.001;
            std::cout << "Time taken: " << duration << "us (" << ms << "ms)\n";
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_timepoint, end_timepoint;
    };
}