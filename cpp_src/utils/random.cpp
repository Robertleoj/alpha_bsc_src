#include "./random.h"
#include <random>

namespace utils {
    double random_double() {
        return (double)(std::rand()) / (double)(RAND_MAX);
    }

    double random_double_range(double mn, double mx) {
        return (mx - mn) * random_double() + mn;
    }
    
    double normalized_double(){
        return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    std::vector<double> random_nums(int n, float mn, float mx){

        std::vector<double> vc;
        for(int i = 0; i < n; i++){
            vc.push_back(random_double_range(mn, mx));
        }
        return vc;
    }

    std::vector<double> multinomial(int n){
        std::vector<double> freqs = random_nums(n, 0, 100);
        
        double sm = 0;

        for(int i = 0; i < n; i++){
            sm += freqs[i];
        }
        
        for(int i = 0; i < n; i++){
            freqs[i] /= sm;
        }

        return freqs;
    }
}