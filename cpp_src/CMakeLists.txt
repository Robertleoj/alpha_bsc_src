cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(AlphaBSc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -O3 -Wextra -pthread") #-DNDEBUG
# set(CMAKE_CXX_FLAGS " -O3") #-DNDEBUG


set(LIBTORCH ".libs/libtorch/")
set(MARIADB_C ".libs/mariadb_conn_c")
set(MARIADB_CPP ".libs/mariadb_conn_cpp")

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH};${MARIADB_C};${MARIADB_CPP}")

include_directories(".libs/libtorch/include")

find_package(Torch REQUIRED)
#find_package(mariadbcpp REQUIRED)


include_directories(".libs/mariadb_conn_c/include") #path to include folder
include_directories(".libs/mariadb_conn_cpp/include") #path to include folder
include_directories(".libs/gsl/include") #path to include folder

link_directories(./.libs/gsl/lib)
link_directories(./.libs/mariadb_conn_cpp/lib/mariadb)

add_library(mariadbcpp STATIC IMPORTED)

set_property(TARGET mariadbcpp PROPERTY IMPORTED_LOCATION "/home/gimli/AlphaBSc/alpha_bsc_src/cpp_src/.libs/mariadb_conn_cpp/lib/mariadb/libmariadbcpp.so") #path to libmariadbcpp.so


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
)

target_link_libraries(self_play
    ${TORCH_LIBRARIES}
    mariadbcpp
    gsl 
    gslcblas
)

