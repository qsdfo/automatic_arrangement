# LOP

## Requirements
You need to download the toolbox <https://github.com/aciditeam/acidano> and add it to your python path. To do this, you can either :
- place the acidano folder in the lop/Source 
- or type in a shell
        
        PYTHONPATH=”${PYTHONPATH}:/path/to/acidano/folder”


## A scenario
### Building the database
The first step is to build the database, i.e. read a corpus of midi file in an orchestra and a piano matrices, split them into train/valid/test matrices and save them in a folder (for instance lop/Data) as .npy files. This is done by executing :

    cd Source/Database; python build_data.py

The train/valid/test split is defined through indexing files, whose path is passed as an argument of the main function in *build_data.py*.
A database can be found at <https://qsdfo.github.io/LOP/database/LOP_database_06_09_17.zip>. \
We simplify orchestrations by mapping rare instruments to more common ones through the *simplify_instrumentation* table located in *build_data_aux.py* that you might want to edit.

### Training
Is made by calling

    cd Source; python main_random_search.py

Parameters at the very beginning of the file are used for execution context and should not be touched, except DEFINED_CONFIG and CONFIG_ID

    N_HP_CONFIG = 1      # In case you run hyper-parameter search
    LOCAL = True            # Was used for Guillimin/non-Guillimin, should be set to True
    DEBUG = False          
    CLEAN = False            # Remove empty folders in the result folders. If you want to run several configs in parallel, set it to False
    DEFINED_CONFIG = True    # Choose between hyper-parameter search, or run for a particular configuration.
    CONFIG_ID = '/13'       # Name of the folder in which results will be stored. If you choose an  existing folder, raise an error. Useless for Hyper-parameter runs.

#### Training parameters
The training parameters (model, optim method, granularity, type of units and quantization) are defined in the commands table :

    commands = [
        'FGgru',
        'adam_L2',
        'event_level',
        'binary',
        '100'           # Set to 100 when using event_level
    ]

#### Defining a specific configuration for a model
DEFINED_CONFIG allows to run a configurations defined in the dictionnary *configs* at the top of *main_random_search.py*.

### Generating
Call

    python result_folder_main.py

The main function of this script takes as input a result folder's path and generate trained weights' plots, midi files (short or whole track), statistics...

## Future
- Adapt to Scikit framework
- Adapt to scikit-multilearn

<!-- # Models
## Difference between RBM based models and LSTM based or mixed
LSTM based models and mixed models can not be initialized with a sequence.
Thus, the inference task is much more difficult for them. In another way, it gives a much powerfull model (no initialization)

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
 -->