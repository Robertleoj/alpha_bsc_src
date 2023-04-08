cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(AlphaBSc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Ofast -O3 -Wextra -pthread -pipe -march=native")# -DNDEBUG")
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
    utils/timer.h
    utils/thread_pool.h
    utils/uuid.h
    utils/uuid.cpp
    games/move.h
    games/game.h
    games/breakthrough.h
    games/breakthrough.cpp
    games/connect4.h
    base/board.h
    MCTS/agent.cpp
    MCTS/agent.h
    MCTS/mc_node.cpp
    MCTS/mc_node.h
    MCTS/mc_tree.cpp
    MCTS/mc_tree.h
    vanilla_MCTS/v_agent.h
    vanilla_MCTS/v_agent.cpp
    vanilla_MCTS/v_mc_node.h
    vanilla_MCTS/v_mc_node.cpp
    vanilla_MCTS/v_mc_tree.h
    vanilla_MCTS/v_mc_tree.cpp
    NN/connect4_nn.h
    NN/connect4_nn.cpp
    NN/nn.h
    NN/nn.cpp
    NN/breakthrough_nn.h
    NN/breakthrough_nn.cpp
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

add_library(common_sources OBJECT
    ${FILES}
)

add_executable(self_play
    self_play.cpp
)

target_link_libraries(self_play
    PRIVATE common_sources
)


add_executable(play_agent
    play_agent.cpp
)

target_link_libraries(play_agent
    PRIVATE common_sources
)


add_executable(eval_agent
    eval_agent.cpp
)

target_link_libraries(eval_agent
    PRIVATE common_sources
)

add_executable(test
    test.cpp
)

target_link_libraries(test
    PRIVATE common_sources
)

if(NOT USERNAME STREQUAL "gimli")


    # add_library(pybinds_sources OBJECT
    #     ${FILES}
    # )

    execute_process(
        COMMAND "python3" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})

    find_package(pybind11 REQUIRED)

    pybind11_add_module(player 
        pybinds.cpp
        ${FILES}
    )

    # target_link_libraries(player PRIVATE common_sources)

    target_compile_options(player PRIVATE -fPIC)

endif()


