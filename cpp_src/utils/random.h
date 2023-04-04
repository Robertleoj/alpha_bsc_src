#pragma once
#include <vector>
#include <map>
#include <unordered_map>


namespace utils {
    double random_double();
    
    double normalized_double();

    double random_double_range(double mn, double mx);

    std::vector<double> random_nums(int n, float mn, float mx);

    std::vector<double> multinomial(int n);
    int sample_multinomial(const std::vector<double>& probs);


    // multinomial sample from map<T, double>
    template <typename T, typename Pmap> 
    T sample_multinomial(const Pmap& probs) {

        std::vector<double> p;
        for (auto& kv : probs) {
            p.push_back(kv.second);
        }
        int idx = sample_multinomial(p);
        auto it = probs.begin();
        std::advance(it, idx);
        return it->first;
    }

    // normalize map<T, K> 
    template <typename T, typename K, typename Pmap>
    std::unordered_map<T, double> softmax_map(Pmap& probs) {
        
        double sum = 0;
        for (auto& kv : probs) {
            sum += kv.second;
        }

        auto new_map = std::unordered_map<T, double>();

        for (auto& kv : probs) {
            new_map[kv.first] = ((double)kv.second) / sum;
        }

        return new_map;
    }

    // multinomial argmax from map<T, double>
    template <typename T, typename K, typename Pmap> 
    T argmax_map(const Pmap& probs) {

        [[maybe_unused]] T best_elem;
        K best_prob = -1;

        for (auto& kv : probs) {
            if (kv.second > best_prob) {
                best_prob = kv.second;
                best_elem = kv.first;
            }
        }

        return best_elem;
    }
}