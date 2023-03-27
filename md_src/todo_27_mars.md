

- Gradually lower temperature in softmax when choosing moves in the opening. 
    - need to make a separate function for this, as we use the regular normalization function to create the target distribution for the neural network.
