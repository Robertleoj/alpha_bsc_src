import player
from sys import argv

p = player.Connect4PerfectCompetitor(1)

if argv[1] == 'white':
    turn = True
else:
    turn = False

assert argv[2] in ('random', 'perfect')
random = argv[2] == 'random'


while True:
    if turn:
        move = input('Move:')
        p.update([move])
    else:
        print(f"Agent move: {p.make_and_get_moves()[0]}")

    turn = not turn
