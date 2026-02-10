from .lagging import LaggingTransformMethod
from .stateful_lagging import StatefulLaggingTransformMethod

"""The lagging history transform. This method theoretically allows the track to
be fed to the network as a sequence of segmentation lines and the track 
predicts the vehicle position and velocity as it goes. This, means a RNN/LSTM
network could be used to predict track data which allows the network to 
theoritcally remember the whole track history such that many previous corners
can be taken into account that would be well outside the input context given
to the class windowing technique. 

Additionally, the windowing technique means that each segmentation line is 
given to the network many many times, whereas this method, for stateful 
networks, means a segmentation line is only repeated onces. Which means the
whole network should perform many many times faster than the classic approach.

The lag is applied to the sequence by shifting the output backward such 
that the first normal, predicts for normals in the past. This allows the
network to obtain information for the upcoming track before it predicts the
output for that normal.

Both the input and output can be patched such that multiple seg. lines are 
passed to the network at once and predicted for. This means the network 
calculates for fewer samples and it means recurrent networks gets more info at
once.

Time to vec can be enabled, which spliced the relative position of the track 
into the encoding. This enables the network to know how far through the track
it is."""