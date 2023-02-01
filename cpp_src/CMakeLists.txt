cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(AlphaBSc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -O3 -DNDEBUG -Wextra")

set(LIBTORCH ".libs/libtorch/")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH}")

# include_directories(".libs/libtorch/include")

# set(Torch_DIR ".libs/libtorch/share/cmake/Torch")
# set(Caffe2_DIR ".libs/libtorch/share/cmake/Caffe2")
find_package(Torch REQUIRED)
# set(DCMAKE_PREFIX_PATH ./.libs/libtorch)
# target_link_libraries(AlphaBSc "${TORCH_LIBRARIES}")

add_executable(AlphaBSc
        main.cpp
        base/types.h    
        base/bitboard.h
        utils/utils.h
        utils/random.h
        utils/random.cpp
        games/move.h
        games/game.h
        games/breakthrough.h
        games/breakthrough.cpp
        games/connect4.h
        games/connect4.cpp
        misc/perft.h
        misc/perft.cpp 
        base/board.h
        MCTS/agent.cpp
        MCTS/agent.h
        MCTS/mc_node.cpp
        MCTS/mc_node.h
        MCTS/mc_tree.cpp
        MCTS/mc_tree.h
        NN/connect4_nn.h
        NN/nn.h
)

target_link_libraries(AlphaBSc PRIVATE -lm
        -Wl,--whole-archive "${TORCH_LIBRARIES}"
        -Wl,--no-whole-archive
        -lpthread
        ${CMAKE_DL_LIBS})


# target_link_libraries(AlphaBSc "${TORCH_LIBRARIES}")
