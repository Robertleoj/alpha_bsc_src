
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
<!-- New slide -->

# Abstract

AI & Machine learning is a very hot topic today, and a recent major success is the AlphaZero algorithm, which reaches superhuman levels in abstract board games such as chess and go only given the rules of the games.
<!-- New slide -->

However, the algorithm requires a massive amount of computing resources to train. We propose a method to reduce the computational cost of training an AlphaZero-like agent with a method we call Late-to-early simulation focus, and we show that our method improves the performance of the agent in two games: Connect4 and Breakthrough.

<!-- New slide -->

# Introduction

AlphaZero is a deep reinforcement learning algorithm that learns to play abstract board games by playing against itself, often reaching superhuman playing strength. The amazing thing about this approach is that it requires no pre-coded expert domain knowledge nor human involvement during the learning process. However, this approach comes at a cost, particularly access to massive computing resources. For example, AlphaZero played 40 million chess games with the help of 5,000 special-purpose computing units (TPUs) to reach state-of-the-art performance in the game of chess.

The most computationally time-consuming component of the learning process is the self-play, that is, playing the games that the agent learns from. We call these games the *training data*. So, a natural question is whether one can achieve similar expertise by using significantly less computing resources.

<!-- New slide -->

Several avenues of research have addressed this, but the one we are concerned with is more efficient generation and use of the training data. 

This is our main contribution -- an improved strategy for generating training data via self-play, resulting in higher-quality training samples, especially in earlier training phases. We call the new strategy Late-To-Early Simulation Focus, abbreviated as LATE.

The method seeks inspiration from curriculum learning, a training strategy that trains machine learning models from more straightforward to more complex examples, thereby imitating the learning order in human curricula. 

<!-- New Slide -->

In early training, the approach emphasizes training experiences gathered from the late stages (i.e., endgame) of individual games. However, as the training progresses, it expands its focus to consider entire games. 

<!-- New Slide -->

In our test domains, agents using our approach for training reach superior playing strength faster than counterpart agents using a standard training approach, requiring significantly less training time. 


We also empirically show and quantify the quality improvement in the training data when using our new approach and compare its effectiveness to that of several other recently published approaches.

<!-- New Slide -->

The presentation's organization is as follows. 
- Background (AZ, other methods)
- Methods (LATE)
- Results
- Conclusion and future work

<!-- New Slide -->

# Background

## Overview
Here we provide the necessary preliminaries. We start by giving an overview of the workings of AlphaZero-style game-playing agents and their training, followed by a summary of recent related approaches.

<!-- New Slide -->

## AlphaZero-style game-playing agents

At a high level, AlphaZero uses a neural-network guided Monte-Carlo Tree Search to choose moves.

<!-- New Slide -->


### Neural network
The neural network function accepts a state as input, and outputs both a value estimate of the state, and a probability distribution over the possible moves in the state, which we call the policy prior.


The value estimate is the neural-network's prediction of the probability of winning the game and the policy prior predicts which moves are likely to be good.


<!-- New Slide -->
### MCTS

Now we'll show how the MCTS is performed with the help of the neural network. 

The algorithm builds a tree, where nodes in the tree are board states and the edges are actions (or moves). The root node is the current board state, and each node stores a value estimate and a policy prior.

<!-- New Slide -->

At first, the tree consists only of the root node, and is then expanded by performing *simulations*.

A simulation consists of three phases: selection, evaluation-and-expansion, and back-propagation.

<!-- New Slide -->

In the selection phase, we traverse down the tree, where in each node we choose the action that maximizes this formula, where the policy prior is used.
> point at this formula, and where the policy prior is in it
$$
    PUCT(s, a, s') = V(s') + c_{{PUCT}} \cdot p_{\boldsymbol{\theta}}(s, a) \frac{\sqrt{N(s)}}{N(s') + 1},
$$
The selection phase ends when we reach a node that is not in the tree, or is a terminal node, that is, a node where the game ends.

<!-- New Slide -->

In the evaluation-and-expansion phase, we evaluate the node we reached in the selection phase using the neural network. We add the node to the tree, and store the value estimate and policy prior in the node.

<!-- New Slide -->

In the back-propagation phase, we update the value estimate of all the nodes we visited in the selection phase. We do this by propagating the value estimate of the node we reached in the evaluation-and-expansion phase up the tree.

<!-- New Slide -->

After some number of simulations, we stop the search and choose the move corresponding to the root-child with the most visits.
> TODO: Maybe put a new slide here if we can afford to add a slide

<!-- New Slide -->
### Training
The neural network is trained using data gathered during agents' self-play. 

In each generation, an agent plays a fixed number of games against itself.

We store each state in each game, and label them with the game's outcome, and the normalized MCTS visit counts of the available moves. This is stored in a so-called replay-buffer.

This results in labeled samples for training the next-generation network.

<!-- New Slide -->

This process continues in a cycle:

Self-play generates data which is stored in the replay buffer
We sample from the replay buffer during training.

Training produces the next-generation neural network, which is then used in self-play, and the cycle repeats.

For those familiar with machine learning, the training objective is to minimize this loss function
$$
    L(\boldsymbol{\theta}, s, z, \boldsymbol{\pi}) = (v_{\boldsymbol{\theta}}(s) -  z)^2 - \boldsymbol{\pi}^\top \log(\boldsymbol{p}_{\boldsymbol{\theta}}(s)) + c||{\boldsymbol{\theta}}||^2
$$
- $\theta$ nn params
- $s$ state
- $z$ game outcome
- $\pi$ normalized MCTS visit counts

In early generations, valuable signals from the training games may be sparse as the agents act more or less randomly. As the agents improve the games become more representative of expert-level play. 

<!-- New Slide -->

<!-- The premise behind our method, as well as the others we review, is avoiding spending much computing resources on generating and training with samples having low-quality target signals. -->

## Related work
There are many methods that try to improve the AlphaZero algorithm. The ones that we look at are the following.

The first is Dynamic Training Window (DTW), which is used in OLIVAW, an AlphaZero-style agent. In DTW, instead of using a fixes-size window, it gradually increases in size as training progresses. This is done to avoid using training samples from early generations, which are of low quality, while still maintaining data diversity in later generations.

The second improvement is Gradually Increasing Simulations (GIS), which also comes from OLIVAW. In GIS, the number of MCTS simulations performed per state during self-play starts small, and is gradually increased as training progresses. This is done to avoid wasting simulations in early generations, where the agent is not very good, and the simulations are not very useful.

The third improvement is Randomized Playout Cap (RPC), which is used in KataGo, another AlphaZero-style agent. In RPC, the number of MCTS simulations performed per move are either very few, or many. Only states where many number of simulations are performed are used for training. The reduced average number of simulations per move allows us to play more games per generation. The improvement is designed to add more diversity to the training data.


<!-- New Slide -->
# Methods

In early training generations, the gameplay is poor, with both sides making many mistakes. Because of this, the target labels are inaccurate and only loosely related to the actual merits of moves and states. 

However, the gameplay generally improves quickly between early training generations, improving the target labels. 

The previous strategies capitalize on this. 

However, another factor is also at play -- the labeling quality also differs within a game. Our method capitalizes on this.

Now, the think-ahead process of MCTS can reliably look some moves ahead. So, when close to terminal states, it quickly zooms in on the correct moves, resulting in the value and policy target labels both being reasonably accurate.

As a result, the labels tend to be significantly more accurate in later game phases than earlier ones. 

> Maybe talk about image as an example, depending on time

We confirm this general phenomenon in the results section. This observation is a central premise behind the enhancement we propose.


<!-- New Slide -->
## LATE

In the Late-To-Early Simulation Focus (LATE) enhancement, we vary the number of MCTS simulations performed during self-play based on *the current training generation and the number of moves played within the game*, according to a weight function.

> Point at the plot of the weight function
> Also the very cool and nice gif and explain.


<!-- New Slide -->

In addition to controlling the number of playouts, the weight also controls the contribution of the training samples to the neural network's learning. That is, samples with low weight contribute less to the training of the neural network. 

For those versed in machine learning, this is done by weighting the loss function with the weight function.

This method avoids wasting much search on generating the noisy training data that is prevalent in the early phases in the game. Additionally, noisy samples are assigned less weight in the training process, further reducing their impact.

As training progresses, the weight function gradually shifts the search effort towards the start of the game, as the training data becomes more accurate.

<!-- New Slide -->

# Results
Now we will go over our research methodology and empirical results.

<!-- New Slide -->

## Games
We use the two-player board games *Connect4* and *Breakthrough* as our test domains.

### Connect4
Connect4 is a well known game, depicted here. 

The main reason for us including this game in our testbed is that it is strongly solved --- we know the actual outcome of all possible game states. Having this information as ground truth allows us to systematically compute and compare the quality of both the models and training samples generated by different self-play strategies. 

<!-- New Slide -->

### Breakthrough
The other game we use is Breakthrough. It is played on a chess-like board.

The initial board state can be seen here. As in chess, white plays first. The pieces move one square straight or diagonally forward, but only capture diagonally. The goal of the game is to get one of your pieces to the opponent's back rank, or to capture all the opponent's pieces.

The strategic complexity of the game is rich enough to require non-trivial strategies to play well while simultaneously being small enough to allow us to train an expert-level agent in a reasonable time frame using moderate computing resources.

<!-- New Slide -->
## Quality assessment of Training data

Since Connect4 is solved, we can compare the value target labels of the training data to the outcome assuming optimal play.

> Show plots

These plots show the RMSE error of the value labels of the training data generated during Connect4 training for both the default and LATE agent. 

The different curves show the error averaged over different training generations. We see that, overall, the error reduces with further training. 

The most important thing to notice here is the effect of the error reducing towards later game stages. 

By contrasting the left and right graphs, we see that the quality of the training samples generated by the LATE agent seems at least as good as the default agent despite running much fewer simulations in early game stages. 

Moreover, because of the sample weighing, the training of the enhanced agents places less emphasis on the (more erroneous) samples in the early game stages. 

This results in more effective training with respect to playing strength, as we see later.

<!-- New Slide -->
## Quality assessment of neural networks
How do the errors in the training data affect the quality of the neural network and the agents' decisions? 

To measure this we created a dataset of 10,000 random realistic Connect4 positions, and evaluated the value estimate of the default and LATE agents on these positions.

> Show twin plots

These plots depict the RMSE error of the network's value output of positions from the dataset, plotted as a function of training generation (left) and the number of simulations (right). 

Each agent has two versions, one using the default training-window size of sampling from the past 20 generations and the other using twice as large a window. 

We see that the network's value output quality is much better in the LATE agent than the default one, both given the same number of training samples (left) and the same computational effort (right). 

Additionally, we see that the LATE agent benefits from a larger window, unlike the default agent. 

It is typically detrimental to use training data from much earlier generations, as it has lower overall quality; however, because of the importance weighing the LATE agent uses in training, it seems it can still benefit from better quality endgame samples from early generations while not being distracted by the more erroneous early-game samples.

<!-- New Slide -->
## Comparison to Other Methods
Now we assess the quality of our approach in contrast to other enhancements which we explained earlier. 

> Insert figures here

These plots are similar to the previous ones, but here the different lines are the different methods.

Again, the left plots show the error as a function of training generation, and the right ones show it as a function of simulations, or computational effort. 

As you can see the LATE method is in a class of its own regarding the value accuracy.

As we will see later, this translates directly into improved playing strength. For fairness, this result should not be interpreted too literally as all methods mostly use the default hyper-parameters and could potentially improve by careful tuning. Nonetheless, the results show the promise of the LATE method.


<!-- New Slide -->
## Playing Strength: Connect4
Now we explore and compare the playing strength difference between the methods, both in Breakthrough and Connect4.

In connect4, all agents play against an optimal agent, and we measure their strength against it throughout training. If the optimal agent can win, it always does so fast as possible, otherwise it puts up the most prolonged resistance possible. Seeing how they fare in this competition gives us a good idea of their playing strength.

Each match consists of 50 games, where each player plays both sides of 25 random openings.

> Insert figure

Keep in mind that it is not possible to get over $50\%$ win rate against the optimal agent.

We see that the LATE agents learn much faster than the others and, interestingly, are the only ones that, in the end, converge to playing on an (almost) equal level to the optimal agent. The LATE agent that uses the larger generation window holds a slight edge over its counterpart. 

The playing strength results are in harmony with the model quality results we presented earlier; after only a few simulations, the LATE agents reach greater playing strength than all the other agents and maintain that advantage throughout training. 

Even very early in their training, the LATE agents have reached a level of playing strength that the other agents have difficulty matching even at the end of their training.

<!-- New Slide -->
## Playing strength: Breakthrough
Now we move on to Breakthrough. Here, an optimal agent is not possible, so we instead match all the agents against the 100th generation of the default agent, which is very strong. We do not have an objective measure of its strength, but it consistently beats expert-level chess players with both colors.

In each match, we use a similar dataset as before but now with 50 openings.

> Insert figure


As we can see, the LATE agents again learn significantly faster than the others. Specifically, the agents require only around $20\%$ of the default agent's training resources to match its performance. It learns twice as fast as its closest rival of other self-play enhancements.

<!-- New Slide -->

# Conclusion and future work
We presented an enhanced scheme, LATE, to expedite self-play learning in AlphaZero-style game-playing agents and evaluated it in two test domains. 

The former, Connect4, is a strongly solved game, which allowed us to continuously evaluate the quality of the training process in various ways, including how accurately the training samples were labeled, how accurate the trained models were, and how accurate the agents' decisions were. 

The latter test domain, Breakthrough, has a much larger state space and is strategically more complex than Connect4. 

In both domains, the LATE enhancement learned far more efficiently than a standard approach, requiring much less computational resources to reach similar or better quality of play. 

Also, unlike the standard approach, a positive side-effect of the enhancement is that it can effectively reuse data from much earlier training generations.

Furthermore, we compared the LATE enhancements with several recently proposed enhancements and found that our approach outperformed the others in our test domains.

<!-- New Slide -->
As for future work, we plan to carry on along three paths. 
- First, we plan to experiment with the proposed enhancement in a broader range of games.

- Second, to make the first plan more easily attainable, we plan to automate the process of selecting various hyper-parameters, particularly those involved in computing the weight function for LATE; currently, they are manually selected based on expected game length and the number of training generations. 

- Third, we plan a more thorough empirical comparison with related methods for improving self-play training ---including those discussed here --- where we tune each method more carefully for the domain at hand. This will allow us to determine their relative strengths and weaknesses more conclusively.

tnx bye

# Q&A
