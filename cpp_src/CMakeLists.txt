cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(AlphaBSc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -O3 -Wextra") #-DNDEBUG
# set(CMAKE_CXX_FLAGS " -O3") #-DNDEBUG

set(LIBTORCH ".libs/libtorch/")

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH}")

include_directories(".libs/libtorch/include")

find_package(Torch REQUIRED)

include_directories("/usr/include/mariadb") #path to include folder
add_library(mariadbcpp STATIC IMPORTED)
set_property(TARGET mariadbcpp PROPERTY IMPORTED_LOCATION "/usr/lib/libmariadbcpp.so") #path to libmariadbcpp.so


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
    # misc/perft.h
    # misc/perft.cpp 
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
)

target_link_libraries(AlphaBSc
    ${TORCH_LIBRARIES}
    mariadbcpp
)
