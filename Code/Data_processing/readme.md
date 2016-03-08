# Data processing

## MXML parser
Main script for this is **mxml_parser.py**
It takes as argument a path to a folder containting mxml files, quantization and pitch per isntrument
Ouputs
1. a dictionary containing the regexp used to catch instrument names in the xml file
2. a picklelized dictionary data.p :
    * 'quantization' : used for the whole database
    * 'instru_mapping' : link between indices and instrument
    * 'orchestra_dimension' : unseful to initialize matrices when unpickling data.p
    * 'scores' :
        * 'pianoroll' : basic pianoroll, with intensities
        * 'articulation' : same as pianoroll, note activation stopped 1
                    frame before the next activation, or last 1 frame only if
                    the note is a staccato
        * 'filename'

Each file is parsed a first time to get is total length.
Then, using a sax analyzer (xml.sax), the file is parsed.
Dynamics are taken into consideration.


PROBLEM :
DB manually modified with musescore -> corrupted xml files

## load_data.py
Once a database has been built and stored in .p file, this script runs
### get_data.py
* Read the .p file
* Concatenate the pianoroll
* Remove useless pitch dimensions (**remove_unused_pitch.py**)
* Get event-level indices, according to the temporal_granularity variable
    (possible values : frame_level, event_level, full_event_level)
* Instanciate Shared variable for Theano models
### get_minibatches_idx
Returns three different lists for train, validate and test sets.
Shuffeling option, but this is probably not a good idea for LOP.
Instead, the best option so far seems to be Leave-One-Out option:
* Before shuffling, split the whole dataset in N (~10) parts
* Build N different train/val/test sets with the test set being successively one of the N part
* Thus, we obtain a set of N tvt set For each, build minibatches inside those subsets qith shuffling

Why not shuffling before ? Because i think if we shuffle, the train and validate/test distribution will be exactly the same, since a lot of vectors are redundant in the database.
