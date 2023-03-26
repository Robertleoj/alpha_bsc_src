
1. endgame playouts, program and run

Soft playout count, less in the beginning, more toward the endgame.

We use weights to both determine the weight of samples in the training loss, and the number of playouts. The number of playouts is $w \cdot n$, where $w$ is the weight, and $n$ is the maximum number of playouts.

In the training, we weigh the gradient of the loss by the weight of each example.

The function we use for the weights is
$$
w(x) = \frac{1}{1 + e^{k - x /a}}
$$
where $x$ is the number of moves that have been played in the game. $k$ is a hyperparameter controlling where the curve starts and converges to one. $a$ is a hyperparameter controlling the steepness of the curve, which we set to about $4.4$. The initial $k$ can be about $3.5$, and decrease towards $-10$. However, the function is very close to uniform for $k=-2.2$, in which case $w(x)=0.9$.

2. tune endgame sampling, and run
3. measure data quality

4. Fix eval positions: we only have 2500 positions. Then evaluate everything again.