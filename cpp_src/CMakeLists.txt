# cmake_minimum_required(VERSION 3.22)
project(AlphaBSc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -O3 -DNDEBUG -Wextra")

add_executable(AlphaBSc
        main.cpp
        base/types.h    
        base/bitboard.h
        utils/utils.h
        games/move.h
        games/game.h
        games/game.cpp
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
)
