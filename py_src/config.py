from enum import Enum
class SampleMethod(Enum):
    uniform = 0
    endgame = 1


buffer_generations = 20

batch_size = 4096
value_loss_ratio = 1.0
weight_decay = 1e-4
learning_rate = 1e-4
num_iterations = 5000

sample_method = SampleMethod.endgame

class dynamic_window:
    on = False
    min = 1
    max = 20
    increase_every = 3

class endgame_training:
    p_max = 1
    p_min = 0
    shift = 4
    generation_uniform = 100


# does not affect algorithm, only time
dl_prefetch_factor = 10
dl_num_workers = 8
log_loss_interval = 100