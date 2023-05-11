
# Pointers from galaxy brains

* Assume background in CS

- follow structure of the paper
    - Abstract
    - Introduction
    - Background
    - Methods
    - Results
    - Conclusion and future work
    - Q&A

* Start with an introduction of the project
    - People will get later information in context

* More visual
    - Explain the steps of MCTS with pictures
    - Show endgame playouts visually
    - just always try to be visual

* Look alive
    - Be animated
    - Be excited
    - Be confident

* REHERSE

## Title Slide
Introduce ourselves.
Our project is Expediting Self-Play Learning in AlphaZero-style Agents.

# Abstract

AI & Machine learning is a very hot topic today. A recent major success is the AlphaZero algorithm, which reaches superhuman levels in abstract board games such as chess and go only given the rules of the games.

However, the algorithm requires a massive amount of computing resources to train. We propose a method to reduce the computational cost of training an AlphaZero-like agent with a method we call Late-to-early simulation focus, and we show that our method improves the performance of the agent in two games: Connect4 and Breakthrough.

# Introduction

AlphaZero is a deep-reinforcement learning algorithm that learns to play abstract board games by playing against itself, often reaching superhuman playing strength. One of the main appeals of this approach is that it requires no pre-coded expert domain knowledge nor human involvement during the learning process. However, this approach comes at a cost, particularly access to massive computing resources. For example, AlphaZero played 40 million chess games with the help of 5,000 special-purpose computing units (TPUs) to reach state-of-the-art performance in the game of chess.

The most computationally time-consuming component of the learning process is generating the training data, i.e., the games, via self-play. So, a natural question is whether one can achieve similar expertise by using significantly less computing resources.
Several avenues of research have addressed this, most notably model improvements, and more efficient generation and use of the training data. Here, we are only concerned with the latter. 

Our main contribution is an improved strategy for generating training data via self-play, resulting in higher-quality training samples, especially in earlier training phases. The new strategy, which we call Late-To-Early Simulation Focus, abbreviated LATE, 

The method seeks inspiration from curriculum learning, a training strategy that trains machine learning models from more straightforward to more complex examples, thereby imitating the meaningful learning order in human curricula. 

In early training, the approach emphasizes training experiences gathered from the late stages (i.e., endgame) of individual games. However, as the training progresses, it expands its focus to consider entire games. In our test domains, agents using our approach for training reach superior playing strength faster than counterpart agents using a standard training approach, requiring significantly less training time. 

We also empirically show and quantify the quality improvement in the training data when using our new approach and compare its effectiveness to that of several other recently published approaches.

The presentation's organization is as follows. 
- Background (AZ, other methods)
- Methods (LATE)
- Results
- Conclusion and future work

# Background

## Overview
Here we provide the necessary preliminaries. We start by giving an overview of the workings of AlphaZero-style game-playing agents and their training, followed by a summary of recent related approaches.
> Should we include this? 

## AlphaZero-style game-playing agents

At a high level, AlphaZero uses a neural-network guided Monte-Carlo Tree Search to choose moves.


### Neural network
The neural network function accepts a state as input, and outputs both a value estimate of the state, and a probability distribution over the possible moves in the state, which we call the policy prior.

$$
(\boldsymbol{p_\theta}, v_\theta) = f_{\boldsymbol{\theta}}(s).   
$$

The value estimate is the neural-network's prediction of the probability of winning the game and the policy prior predicts which moves are likely to be good.

> Make a nice diagram of the neural network
> Done

### MCTS

Now we'll show how the MCTS is performed with the help of the neural network. 

The algorithm builds a tree, where nodes in the tree are board states and the edges are actions (or moves). The root node is the current board state, and each node stores a value estimate and a policy prior.
> Show base MCTS image

At first, the tree consists only of the root node, and is then expanded by performing *simulations*.

A simulation consists of three phases: selection, evaluation-and-expansion, and back-propagation.
> Show selection phases image
In the selection phase, we traverse down the tree, where in each node we choose the action that maximizes this formula, where the value estimate and policy prior are used.
> point at this formula, but do not explain it
$$
    PUCT(s, a, s') = V(s') + c_{{PUCT}} \cdot p_{\boldsymbol{\theta}}(s, a) \frac{\sqrt{N(s)}}{N(s') + 1},
$$
The selection phase ends when we reach a node that is not in the tree, or is a terminal node, that is, a node where the game ends.

In the evaluation-and-expansion phase, we evaluate the node we reached in the selection phase using the neural network. We add the node to the tree, and store the value estimate and policy prior in the node.
> Show evaluation-and-expansion phase image

In the back-propagation phase, we update the value estimate of all the nodes we visited in the selection phase. We do this by propagating the value estimate of the node we reached in the evaluation-and-expansion phase up the tree.
> Show back-propagation phase image

After some number of simulations, we stop the search and choose the move corresponding to the root-child with the most visits.


### Training

The neural network is trained using data gathered during agents' self-play. 

Initially, an agent plays a fixed number of games against itself using an untrained network. For each game, all occurring states are recorded and labeled with the game outcome, and the normalized MCTS visit counts of the available actions.

This results in labeled samples for training the next-generation network for the agent. This process repeats and with each generation the agent should improve in playing strength.

For those familiar with machine learning, the training objective is to minimize this loss function
$$
    L(\boldsymbol{\theta}, s, z, \boldsymbol{\pi}) = (v_{\boldsymbol{\theta}}(s) - z)^2 - \boldsymbol{\pi}^\top \log(\boldsymbol{p}_{\boldsymbol{\theta}}(s)) + c||{\boldsymbol{\theta}}||^2
$$
> put in hints of what the variables are

The training process keeps relevant training samples in a so-called replay buffer during training, from which it samples batches for updating the network. The buffer stores training samples from one or more previous generations. 
> Visualize the buffer?

In early generations, valuable signals from the training games may be sparse as the agents act more or less randomly. As the agents improve the games become more representative of expert-level play. The premise behind our method, as well as the others we review, is avoiding spending much computing resources on generating and training with samples having low-quality target signals.

## Related work
There are many methods that try to improve the AlphaZero algorithm. The ones that we look at are the following.

The first is Dynamic Training Window (DTW), which is used in OLIVAW, an AlphaZero-style agent. In DTW, instead of using a fixes-size window, it gradually increases in size as training progresses. This is done to avoid using training samples from early generations, which are of low quality, while still maintaining data diversity in later generations.

The second improvement is Gradually Increasing Simulations (GIS), which also comes from OLIVAW. In GIS, the number of MCTS simulations performed per state during self-play starts small, and is gradually increased as training progresses. This is done to avoid wasting simulations in early generations, where the agent is not very good, and the simulations are not very useful.

The third improvement is Randomized Playout Cap (RPC), which is used in KataGo, another AlphaZero-style agent. In RPC, the number of MCTS simulations performed per move are either very few, or many. Only states where many number of simulations are performed are used for training. The reduced average number of simulations per move allows us to play more games per generation. The improvement is designed to add more diversity to the training data.


# Methods

In early training generations, the gameplay is poor, with both sides making many mistakes. Because of this, the target labels are inaccurate and only loosely related to the actual merits of moves and states. 

However, the gameplay generally improves quickly between early training generations, improving the target labels. 

The previous strategies capitalize on this. 

However, another factor is also at play -- the labeling quality also differs within a game. Our method capitalizes on this.

The think-ahead process of MCTS can reliably look some moves ahead. 

So, when close to terminal states, it quickly zooms in on the correct moves, resulting in the value and policy target labels both being reasonably accurate.

As a result, the labels tend to be significantly more accurate in later game phases than earlier ones. We confirm this later in this presentation. This observation is a central premise behind the enhancement we propose.


## LATE

In the Late-To-Early Simulation Focus (LATE) enhancement, we vary the number of MCTS simulations performed during self-play based on *the current training generation and the number of moves played within the game*, according to a weight function.

> Point at the plot of the weight function
> Also the very cool and nice gif and explain.

In addition to controlling the number of playouts, the weight also controls the contribution of the training samples to the neural network's learning. That is, samples with low weight contribute less to the training of the neural network. 

For those versed in machine learning, this is done by weighting the loss function with the weight function.


This method avoids wasting much search on generating the noisy training data that is prevalent in the early phases in the game. Additionally, noisy samples are assigned less weight in the training process, further reducing their impact on the training process.


As training progresses, the weight function gradually shifts the search effort towards the start of the game, as the training data becomes more accurate.


# Results
Now we will go over our research methodology and empirical results.

## Games
We use the two-player board games *Connect4* and *Breakthrough* as our test domains.

### Connect4
Connect4 is a well known game, depicted here. The players take turns dropping their disks, with the goal being to get four of your disks in a row. The game is a draw if neither player manages this before the grid fills.

The main reason for us including this game in our testbed is that it is strongly solved --- we know the actual outcome of all possible game states. Having this information as ground truth allows us to systematically compute and compare the quality of the training samples generated by different self-play strategies. 

### Breakthrough
The other game we use is Breakthrough. It is played on a chess-like board.

The initial board state can be seen here. As in chess, white plays first. The pieces move one square straight or diagonally forward, but only capture diagonally. The goal of the game is to get one of your pieces to the opponent's back rank, or to capture all of the opponent's pieces. One player always wins, there are no draws.

The strategic complexity of the game is rich enough to require non-trivial strategies to play well while simultaneously being small enough to allow us to train an expert-level agent in a reasonable time frame using moderate computing resources.


## Experimental methodology

Connect4 is a strongly solved game, which allows us to evaluate the quality of the generated training data by comparing the labels of the training data to ground truth. We consider the game outcome, in $(-1, 0, 1)$ assuming optimal play to be the ground truth. Then, given positions and comparison evaluations, we simply measure the RMSE error. 

This allows us to measure the quality of the target value labels in the training data.

Throughout our experiments, we employ the cumulative number of simulations as a proxy to approximate the computational resources expended in training an agent. This approach is justified as the vast majority of computational resources dedicated to training an agent are consumed by the simulations conducted during self-play. Given that the number of simulations executed per generation varies across the different agents we train, the cumulative number of simulations serves as the most suitable and consistent measure.

## Quality Assessment using ground truth
Our ground-truth information for Connect4 allows us to monitor the quality of the generated training data and our agents at different stages during training, gaining additional insights.

### Training data
Here, we look at the quality of the value estimate. 

> Show plots

These plots show the RMSE error of the value labels of the training data generated during Connect4 training for both the default and LATE agent. 

The different curves show the error averaged over different training generations. We see that, overall, the error reduces with further training. 

The most important thing to notice here is the effect of the error reducing towards later game stages. 

By contrasting the left and right graphs, we see that the quality of the training samples generated by the LATE agent seems at least as good as the default agent despite running much fewer simulations in early game stages. 

Moreover, because of the sample weighing, the training of the enhanced agents places less emphasis on the (more erroneous) samples in the early game stages. 

This results in more effective training with respect to playing strength, as we see later.

### Model
How do the errors in the training data affect the quality of the neural network and the agents' decisions? 

To measure this we created a dataset of 10,000 random realistic Connect4 positions, and evaluated the value estimate of the default and LATE agents on these positions.

> Show twin plots

These plots depict the RMSE error of the network's value output of positions from the dataset, plotted as a function of training generation (left) and the number of simulations (right). 

Each agent has two versions, one using the default training-window size of sampling from the past 20 generations and the other using twice as large a window. 

We see that the network's value output quality is much better in the LATE agent than the default one, both given the same number of training samples (left) and the same computational effort (right). 

Additionally, we see that the LATE agent benefits from a larger window, unlike the default agent. 

It is typically detrimental to use training data from much earlier generations, as it has lower overall quality; however, because of the importance weighing the LATE agent uses in training, it seems it can still benefit from better quality endgame samples from early generations while not being distracted by the more erroneous early-game samples.

## Comparison to Other Methods
Now we assess the quality of our approach in contrast to other enhancements which we explained earlier. 

> Insert figures here

These plots are similar to the previous ones, but here the different lines are the different methods.

Again, the left plots show the error as a function of training generation, and the right ones show it as a function of simulations, or computational effort. 

The upper plots are the error of the network output as before, but the lower plots show the error after 600 MCTS simulations.

As you can see the LATE method is in a class of its own regarding the value accuracy of both the neural-network model and the MCTS.

As we will see later, this translates directly into improved playing strength. For fairness, this result should not be interpreted too literally as all methods mostly use the default hyper-parameters and could potentially improve by careful tuning. Nonetheless, the results show the promise of the LATE method.


## Playing strength
Now we explore and compare the playing strength difference between the methods, both in Breakthrough and Connect4.

### Connect4
In connect4, all agents play against an optimal agent, and we measure their strength against it throughout training. If the optimal agent can win, it always does so fast as possible, otherwise it puts up the most prolonged resistance possible. Seeing how they fare in this competition gives us a good idea of their playing strength.

Each match consists of 50 games, where each player plays both sides of 25 random openings.

> Insert figure

Keep in mind that it is not possible to get over $50\%$ win rate against the optimal agent.

We see that the LATE agents learn much faster than the others and, interestingly, are the only ones that, in the end, converge to playing on an (almost) equal level to the optimal agent. The LATE agent that uses the larger generation window holds a slight edge over its counterpart. 

The playing strength results are in harmony with the model quality results we presented earlier; after only a few simulations, the LATE agents reach greater playing strength than all the other agents and maintain that advantage throughout training. 

Even very early in their training, the LATE agents have reached a level of playing strength that the other agents have difficulty matching even at the end of their training.

### Breakthrough
Now we move on to Breakthrough. Here, an optimal agent is not possible, so we instead match all the agents against the 100th generation of the default agent, which is very strong. We do not have an objective measure of its strength, but it consistently beats expert-level chess players with both colors.

In each match, we use a similar dataset as before but now with 50 openings.

> Insert figure


As we can see, the LATE agents again learn significantly faster than the others. Specifically, the agents require only around $20\%$ of the default agent's training resources to match its performance. It learns twice as fast as its closest rival of other self-play enhancements.



# Conclusion and future work
We presented an enhanced scheme, LATE, to expedite self-play learning in AlphaZero-style game-playing agents and evaluated it in two test domains. 

The former, Connect4, is a strongly solved game, which allowed us to continuously evaluate the quality of the training process in various ways, including how accurately the training samples were labeled, how accurate the trained models were, and how accurate the agents' decisions were. 

The latter test domain, Breakthrough, has a much larger state space and is strategically more complex than Connect4. 

In both domains, the LATE enhancement learned far more efficiently than a standard approach, requiring much less computational resources to reach similar or better quality of play. 

Also, unlike the standard approach, a positive side-effect of the enhancement is that it can effectively reuse data from much earlier training generations.

Furthermore, we compared the LATE enhancements with several recently proposed enhancements and found that our approach outperformed the others in our test domains.

As for future work, we plan to carry on along three paths. 
- First, we plan to experiment with the proposed enhancement in a broader range of games.

- Second, to make the first plan more easily attainable, we plan to automate the process of selecting various hyper-parameters, particularly those involved in computing the weight function for LATE; currently, they are manually selected based on expected game length and the number of training generations. 

- Third, we plan a more thorough empirical comparison with related methods for improving self-play training ---including those discussed here --- where we tune each method more carefully for the domain at hand. This will allow us to determine their relative strengths and weaknesses more conclusively.



tnx bye



# Q&A

## Our Project






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







