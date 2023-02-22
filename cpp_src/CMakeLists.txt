cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(AlphaBSc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -O1 -Wextra -pthread")# -DNDEBUG")
# set(CMAKE_CXX_FLAGS " -O3") #-DNDEBUG


set(LIBTORCH ".libs/libtorch/")
#set(LIBTORCH "/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib64/python3.9/site-packages/torch/share/cmake/")

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH}")
# set(CUDNN "/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib/python3.9/site-packages/nvidia/cudnn/")

#set(CUDNN_LIBRARY_PATH "/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib64/python3.9/site-packages/nvidia/cudnn/lib/libcudnn.so.8")
#set(CUDNN_INCLUDE_PATH "/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib64/python3.9/site-packages/nvidia/cudnn/include")

#include_directories(".libs/libtorch/include")
find_package(Torch REQUIRED)



include_directories(".libs/gsl/include") #path to include folder
# include_directories("/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib/python3.9/site-packages/nvidia/cudnn/include") #path to include folder

link_directories(./.libs/gsl/lib)
#link_directories(/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib/python3.9/site-packages/nvidia/cudnn/lib/)
# 



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Find Torch VERSION: ${Torch_VERSION}")

add_executable(self_play
    main.cpp
    base/types.h    
    base/bitboard.h
    utils/utils.h
    utils/random.h
    utils/random.cpp
    utils/dirichlet.h
    utils/dirichlet.cpp
    games/move.h
    games/game.h
    games/breakthrough.h
    games/breakthrough.cpp
    games/connect4.h
    games/connect4.cpp
    base/board.h
    MCTS/agent.cpp
    MCTS/agent.h
    MCTS/mc_node.cpp
    MCTS/mc_node.h
    MCTS/mc_tree.cpp
    MCTS/mc_tree.h
    NN/connect4_nn.h
    NN/connect4_nn.cpp
    NN/nn.h
    DB/db.h
    DB/db.cpp
    self-play/selfplay.h
    self-play/selfplay.cpp
    config/config.h
    config/config.cpp
    sqlite/sqlite3.h
    sqlite/sqlite3.c
)

target_link_libraries(self_play
    ${TORCH_LIBRARIES}
    gsl 
    gslcblas
    dl
)

