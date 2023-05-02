



# Introduce ourselves


# Open by introducing AlphaZero 
- machine learning algorithm to play games such as chess and go
- learns by playing itself
- beat stockfish in chess after four hours (on google hardware)


# History of AlphaZero
AlphaGo, the predicessor of AlphaZero, beat the world champion in Go and was the first Go program to beat a professional Go player. This was a feat not expected to happen so soon.

The reason is that it is not possible to play Go well by simply searching, because there are simply way too many moves to consider. There is also no easy way to measure how good a position is.

To look ahead just two moves, the computer would need to look at four billion positions! Three moves would take two hundred trillion moves. 

So instead of looking at all possible moves, we need to choose the moves we consider very wisely. This is what AlphaGo did very well.

AlphaZero is an evolution of AlphaGo that can play any two-player turn-taking board game such as Go or Chess, and does not use any human-made knowledge. It only gets the rules of the game.



# The AlphaZero algorithm
- Uses a neural network to evaluate the position and to predict which moves should be considered.
  - Like a human player, it can look ahead to see what the board will look like after a few moves.
  - It looks ahead by simulating moves for both players.
    - Uses the neural net to evaluate the board after each move and updates the prediction of moves to consider.
  - After a certain number of simulated moves, it plays the move that it thinks is the best.

- To learn, the agent plays games against itself.
  - At first, the neural net knows nothing, and has no idea how to play the game.
    - Nearly all the moves are random.
  - However, as it plays more and more games, the neural network is improved.
    - The prediction of the best moves and expected outcome of the game gets more accurate.
    - This leads to better play and more accurate predictions.

# The problem (FUCKING SLOW)
AlphaZero outperformed Stockfish after just four hours of training, but this training was performed with a massive amount of resources. 5000 TPUs (very expensive) were used to play the games, and 64 more were used to train the neural network with the generated games. 

These resources are immense, and not feasible for most universities, companies, or research groups. 

With commodity hardware, it would take very long to train.

The problem mostly lies in the self-play, which is much more expensive than training the neural network using these games. 

# Prior work on the problem (KataGo, OLIVAW, etc)
Many successful attempts have been made to reduce the computational cost of training an AlphaZero-like algorithm. Notable ones are
- KataGo
- OLIVAW


# Specific issue we deal with
We want to exploit two specific properties that arise at the start of training. They both stem from the fact that the neural network is not competent early in training.

## Training data bad early game
Since the agent is so bad, the outcomes of the games generated during self-play are not very representitive of what the outcome would be if the players were strong.

This means that the agent is getting bad information - it wants to learn how good the positions are given strong play. 

Because of this, the agent learns slower.

However, this issue becomes less pronounced as we approach the end of the game. This is because there are fewer chances for bad moves, and more chances of finding winning sequences.

## Search is inefficient in early generations
When the agent is looking ahead (searching) early in training, it both has no idea which moves would be good to consider, nor how good their resulting positions are.
<!-- So what good does it to to look at possible future positions if you have no idea if they are good or not?  -->
Since it has no idea how good positions are, there is not much use in looking many moves ahead. It simply results in the agent aimlessly simulating a lot of moves, without much improvement in its move choice. 

This wastes simulations, and therefore time/computation/money.

# Introduction of experiments
We implemented the algorithm to run our experiments. We will be using two games for our experiments. 

## Connect4
The first game we use is Connect4. We chose it because it is a very simple game, and has been strongly solved, so we can always know the best move in a position, and the outcome assuming both players play optimally.

Thus, when we train an agent on connect4, we can measure the quality of the training data and the agent's predictions by comparing it to the result from the solver.

> SHOW IMAGE OF CONNECT4 AND EXPLAIN RULES

# Show results of training data measurements
We implemented the algorithm and trained an unmodified agent on Connect4.

To verify the existence of the first problem, we measured the quality of the training data by comparing it to the result from the solver.

Here, we only measure the accuracy of the value target, not the policy target. We get the following results:

> Explain the graphs

As we can see, the accuracy of the value target is very low at the start of training. More interestingly, it is much lower in the early game than in the endgame. Close to the end of the game, the accuracy becomes very high.

This clearly confirms the existence of the first problem. The second problem is then simply a result of the first problem.

# Our proposed solution

We want to try to mitigate these two problems.

The idea of our method is to make the agent learn the endgame first, and then gradually increase its search effort toward the start of the game.

There are a few reasons why this is a good idea:
- The training data we get from self-play is better in the endgame, so the agent learns to predict states later in the game with more accuracy.
- If the agent is stronger in the endgame we can perform more search closer to the endgame, but less search in the early game. This saves us computation.
- This means that we can gradually get better training data for the early game, and gradually increase the search effort in the early game.

The distribution of the search can be seen in this plot:

> Show weight function plot

As can be seen, the search effort is very low in the early game, and gradually increases toward the end of the game, but moves towards the start of the game as training progresses, eventually becoming uniform.

Additionally, while training the neural network, we weigh the contribution of the states with more search effort more. This means that the agent puts more emphasis on the states with more search effort, which are the states closer to the endgame early in training.

# Performance of our method in Connect4

Here we plot the accuracy of the value target and the policy target of both the unmodified agent and our agent over training. We clearly see that our agent learns faster than the unmodified agent.
> Insert evals plot_with_friends

Here we make them compete. The way we chose to do this is to fix the generation 100 model of the unmodified run, and plot the performance of the agent against it throughout training:
> Insert competition plot

# Comparison to other methods
We chose two other improvements to compare to: simply increasing the amount of search over time, and gradually increasing the size of the training window.

Here are all the modificiations together on the same plot:
> Insert evals plot_with_friends with all methods

Here we make them compete:
> Insert competition plot with all methods

# Breakthrough
The second game we use is Breakthrough. We chose it because it is a simple game, but not as simple as Connect4. It is also not strongly solved, so we cannot know the best move in a position, or the outcome assuming both players play optimally.

Thus, when we train an agent on Breakthrough, we can only measure the agent's performance by letting it play itself or another agent.

> SHOW IMAGE OF BREAKTHROUGH AND EXPLAIN RULES


# Breakthrough experiments
To see if the improvement generalizes to a more complex game, we trained the unmodified agent and our agent on Breakthrough.

We measured the performance of the agents by letting them play against each other the same way as in the Connect4 experiments.
> Insert competition plot

As we can see, the improvement is not as drastic as in Connect4, but the agent still learns much faster than 

# Breakthrough: other methods
We also tried the other methods on Breakthrough. Here are the results:

> Insert competition plot with all methods







