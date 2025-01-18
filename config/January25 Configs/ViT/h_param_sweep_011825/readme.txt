- 01/18/25
- This directory contains configs that perform a hyperparameter sweep for SNR-V2-L2*
- This is in response to Vivek's request to implement this method to run a carefull hyperparameter sweep.
- I am starting with a moderately finer sweep of the regularizaion strength for two reasons:
- 1: I have reduced the width, but I am still keeping it somewhat wide since I have only preliminary results on a very small sweep over 4 hyperparameters and a single seed.
- 2: It is finer since we know that in this region, L2 variants perform well and drop off quickly.

- 12 layers with learning rate 1e-3: 

- optimal reg_str was 1e-4 on my "initial" sweep of other L2 methods:
- eta = 0.5, 0.55, 0.6, 0.65, 0.7, 0.8
- 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
- reset_freq = half a task as usual for now
