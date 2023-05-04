
Baseline player is perfect agent in connect4
> Do we have to have random openings then? Because the perfect agent plays a random move given that it is a move with the optimal outcome
> We can also have an agent that always makes the move that prolongs the game the most, which is then deterministic


For randomized cap, the issue is the amount of training data, so quadruple the number of games played in each generation and run connect4 and breakthrough again

Combine the competition plots into one, and make it long, double column in the paper.
> Done
> Should we display confidence intervals for win rates?

average the training data quality in the plot.
> Done
> Training data for default run is very bad - what to do
> Training data for endgame playouts is very good, and improves very stedily. What do?

> We suspect that a learning rate change might affect the non-endgame results, so we should try using a scheduler that mimics what we did in the endgame run.

> Should we truncate all connect4 results at generation 80?
> It would fix endgame_normal_window issues, and data worsening issues
> It keeps the point - that endgame is faster in the beginning

# Rerun:
## High priority
Default with lr scheduler emulating endgame run
Monotone with lr scheduler emulating endgame run
Dynamic with lr scheduler emulating endgame run

Results section



# New questions

Should we plot the training data errors where the error is weighted by the endgame weight - this would justify using a very large window in the training of the endgame agent.

# Points
Keep both endgame20 and endgame40 in plot

Keep endgame40 data quality plot

model quality plot with endgame20&40, and default20&40, we want to show that we can use a large window with endgame, as the data is good

delete breakthrough weight function plot

Split experimental setup into hyperparameters for agent and training

WRITE

Yngvi write:
> assessing training data quality 
> assessing model quality
> experimental methodology


