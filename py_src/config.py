from enum import Enum
class SampleMethod(Enum):
    uniform = 0
    endgame = 1


buffer_generations = 10

batch_size = 4096
dl_num_workers = 8
value_loss_ratio = 2.0
weight_decay = 1e-4
learning_rate = 1e-4
num_iterations = 5000
log_loss_interval = 100

sample_method = SampleMethod.uniform

class dynamic_window:
    on = False
    min = 1
    max = 20
    increase_every = 3

class endgame_training:
    p_max = 0.5
    p_min = 0
    shift = 5
    generation_uniform = 30


# does not affect algorithm, only time
dl_prefetch_factor = 10