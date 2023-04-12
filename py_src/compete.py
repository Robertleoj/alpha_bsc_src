import player
from sys import argv



def main():

    if(len(argv) < 7):
        print("Usage: ")
        print("python3 compete.py <game_name> <playouts> <num_agents> <run_name_1> <gen_1> <run_name_2> <gen_2>")

    game_name = argv[1]
    playouts = int(argv[2])
    num_agents = int(argv[3])
    run_name_1 = argv[4]
    gen_1 = int(argv[5])
    run_name_2 = argv[6]
    gen_2 = int(argv[7])

    p1 = player.Competitor(run_name_1, game_name, gen_1, num_agents, playouts)
    p2 = player.Competitor(run_name_2, game_name, gen_2, num_agents, playouts)

    moves1 = p1.make_and_get_moves()
    print(moves1)
    p2.update(moves1)
    moves2 = p2.make_and_get_moves()
    print(moves2)
    

if __name__ == "__main__":
    main()