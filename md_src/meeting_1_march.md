


# Questions

- $PUCT$ for unexplored nodes


- Found pseudocode from Deepmind that does not use $PUCT$
    - Calculated as:  
      $c_{PUCT} = \log(\frac{N + k_1 + 1}{k_1}) + k_2$,  
      $k_1 = 19652$,  
      $k_2 = 1.25$.  
      So full $PUCT$ is
      $$
      PUCT = V + P\frac{\sqrt{N}}{n + 1}\left(\log\left(\frac{N + k_1 + 1}{k_1}\right) + k_2\right) 
      $$
    - Should we do the same?

- Should we delete the tree after each move? If we don't, the Dirichlet noise will not affect the playouts as much. 
- For how many moves should we sample from $\pi$ in self-play?
- What do the scores mean in the alpha-beta brute force?
- # **`Where server???`**
- Which improvements should we try out? See `improvement_ideas.md`

# Comparing to ground truth
## Policy prior
We can evaluate each child node with $\alpha-\beta$ search, obtaining a ground-truth policy $\pi^*$. Can't we then compare $\pi$ and $P$ to $\pi^*$ with some kind of difference measure?

We know which moves are winning, which ones are drawing, and which ones are losing. We also know how many moves until the outcome with optimal play from both sides. 

We can
* Use the ground truth to create a policy with some kind of softmax and compare
* We can measure the probability mass on the suboptimal moves (losing if wins are possible, drawing if wins and draws are possible, etc.)

## Evaluation
Win, draw, loss is $-1$, $0$, $1$. Compute distance from this to $v$.

