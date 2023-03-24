cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(AlphaBSc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -O3 -Wextra -pthread -pipe -march=native")# -DNDEBUG")
# set(CMAKE_CXX_FLAGS " -O3") #-DNDEBUG

set(USERNAME $ENV{USER})

message("Username is ${USERNAME}")

if(USERNAME STREQUAL "gimli")

    message("On smallvoice. Using python version of libtorch.")

    set(LIBTORCH "/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib64/python3.9/site-packages/torch/share/cmake/")

    set(CUDNN_LIBRARY_PATH "/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib64/python3.9/site-packages/nvidia/cudnn/lib/libcudnn.so.8")

    set(CUDNN_INCLUDE_PATH "/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib64/python3.9/site-packages/nvidia/cudnn/include")

    link_directories(/home/gimli/AlphaBSc/alpha_bsc_src/py_src/.venv/lib/python3.9/site-packages/nvidia/cudnn/lib/)

elseif(USERNAME STREQUAL "cluster")
    
        message("On cluster. Using python version of libtorch.")
    
        set(LIBTORCH "/home/cluster/.local/lib/python3.10/site-packages/torch/share/cmake")

        set(CUDNN_LIBRARY_PATH "/home/cluster/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8")

        set(CUDNN_INCLUDE_PATH "/home/cluster/.local/lib/python3.10/site-packages/nvidia/cudnn/include")

        link_directories(/home/cluster/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/)
    
else()
    message("On local. Using .libs/libtorch.")
    set(LIBTORCH ".libs/libtorch/")

endif()



set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH}")

find_package(Torch REQUIRED)

include_directories(".libs/gsl/include") #path to include folder

link_directories(./.libs/gsl/lib)

link_libraries(
    ${TORCH_LIBRARIES}
    gsl 
    uuid
    gslcblas
    dl
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Find Torch VERSION: ${Torch_VERSION}")

set(FILES 
    base/types.h    
    base/bitboard.h
    utils/utils.h
    utils/utils.cpp
    utils/random.h
    utils/random.cpp
    utils/dirichlet.h
    utils/dirichlet.cpp
    utils/colors.h
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
    simulation/simulation.h
    simulation/evaluation.cpp
    simulation/self_play.cpp
    simulation/structs.h
    simulation/mutual.h
    simulation/mutual.cpp
    simulation/endgame_playouts.h
    config/config.h
    config/config.cpp
    sqlite/sqlite3.h
    sqlite/sqlite3.c
    global.h
)

add_executable(self_play
    self_play.cpp
    ${FILES}
)


add_executable(play_agent
    play_agent.cpp
    ${FILES}
)

add_executable(eval_agent
    eval_agent.cpp
    ${FILES}
)


