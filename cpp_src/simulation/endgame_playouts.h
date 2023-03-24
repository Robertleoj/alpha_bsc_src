#pragma once

#include "../config/config.h"
#include <cmath>
#include <math.h>

struct EndgamePlayoutWeights {
  double shift;
  double slope;

  EndgamePlayoutWeights(double generation) {
    double uniform_at =
        config::hp["endgame_playout_uniform_generation"].get<double>();
    double uni_const =
        config::hp["endgame_playout_uniform_const"].get<double>();
    double base_shift = config::hp["endgame_playout_shift"].get<double>();
    this->slope= config::hp["endgame_playout_slope"].get<double>();
    double power = config::hp["endgame_playout_power"].get<double>();

    double b = config::hp["endgame_playout_b"].get<double>();

    double p = -std::log(b*std::pow(generation / uniform_at, power)+1) / std::log(b);
    this->shift = base_shift * (1 - p) + (uni_const)*p;
  }

  double operator()(double x) { return 1 / (1 + exp(shift - x / slope)); }
};