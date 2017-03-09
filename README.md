# LOP

## Future
- Find better representation
- Categorical (easy)

# Training
## Initialization of the visible units
### Initialization Gibbs chain
Choice : Random uniform
#### Previous frame
Gibbs chain will stay in the init value
#### Bernoulli p = 0.5
Stuck early in the init state
#### Random uniform [0,1]
Good :)

# Optimizer
Rms prop seems to be the best

# Generation
## Threshold on the output probability ?
Probabilities < 0.5 are set to 0
Actually not the case. High number of sampling steps should make this useless.
