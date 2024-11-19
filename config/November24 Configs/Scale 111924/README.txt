In this experiment we increase the model scale to a factor of 256.

Previously, we attempted a scale of 16.

To achieve the scale of 256 we increase the number of heads and neurons by a factor of 16, increasing the non-embedding parameters by a factor of 16^2.

We increase the dataset to B * T * 3 * 256^0.74 tokens which is 95% of the complete works of Shakespeare.

For computational speed, we decrease the number of tasks to 50 from 500 and the number of epochs per task from 100 to 20.

This results in a computation time of 1 hour and 47 minutes. I originally estiamted a computation time of 3 hours. 

The results do not look great. The online/cumulative average/terminal loss taper off at about 3.0, which is quite large. 

L2 outperforms SNR+L2 marginally, by about a negligible amount I would argue. 

I suggest that we increase the number of epochs to 50 and to 100, running a single seed each just for comparison sake. Or, without logging the neuron resets and ages.

The neuron ages take 30 GB and the neuron resets take 23 GB at this scale, so increasing epochs without addressing a save freq will result in memory issues.
