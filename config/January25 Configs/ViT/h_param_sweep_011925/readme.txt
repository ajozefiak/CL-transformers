Here we perform a finer sweep of SNR-V2-L2*

Best training accuracy for SNR-V2-L2*:

For (L3, 1e-3):
reg_str: 5e-05 reset_percentile: 0.5, reset_freq: 0.01667, terminal 50 task accuracies: 0.93457 (0.00746)

Therefore, I am running the following sweep for (L3, 1e-3):
reg_strs = [5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4]
reset_perentiles = [0.45, 0.475, 0.5, 0.55, 0.55, 0.6]
reset_freqs = [1 / (i * epochs * batches) for i in [0.5, 1, 2]]

The 01/20/25 date just ran the 01/19/25 job, but wiht a smaller time limit due to server maintenance