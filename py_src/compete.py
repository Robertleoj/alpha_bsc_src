import player
from sys import argv
import json
from utils import CompetitionResult, CompetitionResultPlayer

from tqdm import tqdm

def play_openings(p1, p2, openings):

    for opening in tqdm(zip(*openings), desc="Playing openings"):
        p1.update(opening)
        p2.update(opening)

def get_openings(game_name):
    with open(f"../db/{game_name}_openings.json", 'r') as f:
        return json.load(f)

def compete(p1, p2, openings):
    play_openings(p1, p2, openings)
    games = [[] for _ in range(len(openings))]

    current_player = 0
    players = [p1, p2]
    tq = tqdm(desc="Competing")
    while True:
        moves = players[current_player].make_and_get_moves()

        # print(f"Agent {current_player} moves: ")
        # print(moves)

        if all([move == '' for move in moves]):
            return games

        for i, move in enumerate(moves):
            if move == '':
                continue
            games[i].append(move)
        
        players[1 - current_player].update(moves)
        current_player = 1 - current_player
        tq.update(1)

def get_agent(run_name, game_name, gen, num_agents, playouts, perfect_random=None):
    if run_name == "perfect":
        assert game_name == 'connect4'
        assert perfect_random is not None
        return player.Connect4PerfectCompetitor(num_agents, perfect_random)
    else:
        return player.Competitor(run_name, game_name, gen, num_agents, playouts)

def compete_result(game_name, playouts, run_name_1, gen_1, run_name_2, gen_2, perfect_random=None):
    openings = get_openings(game_name)

    print(f"Playing {len(openings)} openings on both sides")

    num_agents = len(openings)

    p1 = get_agent(run_name_1, game_name, gen_1, num_agents, playouts, perfect_random)
    p2 = get_agent(run_name_2, game_name, gen_2, num_agents, playouts, perfect_random)

    games1 = compete(p1, p2, openings)

    p1_white_results = p1.get_results()
    p2_black_results = p2.get_results()

    # p1_results = p1.get_results()
    # p2_results = p2.get_results()

    # print("Player 1: ", sum(p1_results)/len(p1_results))
    # print("Player 2: ", sum(p2_results)/len(p2_results))

    print("Switching sides...")

    p1 = get_agent(run_name_1, game_name, gen_1, num_agents, playouts, perfect_random)
    p2 = get_agent(run_name_2, game_name, gen_2, num_agents, playouts, perfect_random)

    games2 = compete(p2, p1, openings)
    
    p1_black_results = p1.get_results()
    p2_white_results = p2.get_results()

    mean_norm = lambda x: (sum(x)/len(x) + 1) / 2

    print(f"Player 1: white={mean_norm(p1_white_results)}, black={mean_norm(p1_black_results)}")
    print(f"Player 2: white={mean_norm(p2_white_results)}, black={mean_norm(p2_black_results)}")

    p1_results = p1_white_results + p1_black_results
    p2_results = p2_white_results + p2_black_results

    print("Player 1: ", mean_norm(p1_results))
    print("Player 2: ", mean_norm(p2_results))

    return CompetitionResult(
        CompetitionResultPlayer(p1_white_results, p1_black_results),
        CompetitionResultPlayer(p2_white_results, p2_black_results)
    )


def main():

    if(len(argv) != 7):
        print("Usage: ")
        print("python3 compete.py <game_name> <playouts> <run_name_1> <gen_1> <run_name_2> <gen_2>")

    game_name = argv[1]
    playouts = int(argv[2])
    run_name_1 = argv[3]
    gen_1 = int(argv[4])
    run_name_2 = argv[5]
    gen_2 = int(argv[6])

    openings = get_openings()

    print(f"Playing {len(openings)} openings on both sides")

    num_agents = len(openings)

    p1 = player.Competitor(run_name_1, game_name, gen_1, num_agents, playouts)
    p2 = player.Competitor(run_name_2, game_name, gen_2, num_agents, playouts)

    games1 = compete(p1, p2, openings)

    p1_results = p1.get_results()
    p2_results = p2.get_results()

    print("Player 1: ", sum(p1_results)/len(p1_results))
    print("Player 2: ", sum(p2_results)/len(p2_results))

    print("Switching sides...")

    p1 = player.Competitor(run_name_1, game_name, gen_1, num_agents, playouts)
    p2 = player.Competitor(run_name_2, game_name, gen_2, num_agents, playouts)

    games2 = compete(p2, p1, openings)
    
    p1_results += p1.get_results()
    p2_results += p2.get_results()

    print("Player 1: ", p1_results)
    print("Player 2: ", p2_results)
    print("Player 1: ", sum(p1_results)/len(p1_results), len(p1_results))
    print("Player 2: ", sum(p2_results)/len(p2_results), len(p2_results))


if __name__ == "__main__":
    main()