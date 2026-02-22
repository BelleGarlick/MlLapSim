# ML LapSim

This repo should have everything required to data the dataset and preprocess it then test it and train on it.
Due to non-complete agreements I will be able to work on this repo soon and therefore will not be able to 
contribute to it. So apologies if you run into issues, I will be unable to help.


## Installation
Once you've cloned this repository, you should be able to, from the directory root run: `./tools/install.sh`.
This will create the environment and install the requirements and package into it. 

To verify it all installed correctly you can run `./tools/test.sh`. 


## Dataset Preprocessing
There are a few steps prior to training that need to be performed, largely pertaining to 
preprocessing the dataset. If you're testing the dataset then you just need to test one whichever
subset you wish. However, if you're training then you'll want to perform these steps on the 
training and test sets.

### 1. Splicing
The first step of preparing the dataset is to create the segmentation lines. This represents the
x1, y1, x2 and y2 ordinates defining the segmentation line as well as the position and velocity
of the vehicle at the point the vehicle passes through the line. Metadata for the vehicle is
also stored in this file.

This step also defines the gap between the segmentation lines. Our model was training on a 10m 
gap. Future work could be done to decrease this distance or more optimal place lines upon the
track. E.g. increasing density of lines at corners/decreasing them on straights.

Each optimal control track designed output by the simulator has its own folder in the dataset. Each
of those folders (which contain the track representation and the optimal control line) will have a
1:1 mapping of folders to spliced files.

To run the splicing run `./tools/cli.sh splice --src <src> --dest <dest> --spacing <spacing>`
e.g. `./tools/cli.sh splice --spacing 10 --src ~/Downloads/DownloadedDataset/test --dest /dataset/spliced/test/`

### 2. Encoding
To encode the spliced track data into a format that the model and train/test on, we need to encode
the data as described in Garlick & Bradley 2022. The encoding function will do this. The encoder
will take the track segmentation lines and position/acceleration and velocity and decompose it into
the inputs and outputs. 

The encoder will output these partitions which are a list of all the encodings of tracks per track,
where each datum comprising that track are grouped together. This provides the tracks in a format 
which make it easy to later create the foresight/sampling windows.

The encoder will, by default, output one encoded track (aka partition) for input track of data.
However, during training or testing, it's easier to load up a partition of encoded data, 
process it, then train on it. Since the full dataset is too large, to load into memory for most 
computers, you can split into a predefined number of partitions which allows you to load into
memory and then train on that subset.

Applying --flip will double the output items, including the normal track encoding and a flipped 
variant of the track. 

To run the splicing run `./tools/cli.sh encode --src <src> --dest <dest>`
e.g. `./tools/cli.sh encode --src ~/Downloads/DownloadedDataset/test --dest /dataset/spliced/test/` 
if you want flipping and partitioning then:
e.g. `./tools/cli.sh encode --src ~/Downloads/DownloadedDataset/test --dest /dataset/spliced/test/ --flip --partitions 10` 

## Training / Testing
In notebooks/ there is a training notebook that will walk you through training the models. You will
have already need to have run the splicer and encoder to run train/testing notebooks.