# LOP

## Future
- Find better representation
- Categorical (easy)

# Models
## Difference between RBM based models and LSTM based or mixed
LSTM based models and mixed models can not be initialized with a sequence.
Thus, the inference task is much more difficult for them. In another way, it gives a much powerfull model (no initialization)

# Optimizer
Rms prop seems to be the best


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

# Generation
## Threshold on the output probability ?
Probabilities < 0.5 are set to 0
Actually not the case. High number of sampling steps should make this useless.
