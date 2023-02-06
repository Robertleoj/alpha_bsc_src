#pragma once
#include <vector>


namespace utils {
    double random_double();
    
    double normalized_double();

    double random_double_range(double mn, double mx);

    std::vector<double> random_nums(int n, float mn, float mx);

    std::vector<double> multinomial(int n);
}