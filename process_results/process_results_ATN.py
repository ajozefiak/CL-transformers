import numpy as np
import matplotlib.pyplot as plt
import pickle 

path_root = '/nobackup1/jozefiak/CL/Results/ATN/'
path_subroots = ['ATN_092324_large',
                'ATN_092324_large_50',
                'ATN_092324_small'
]

epochs_list = [100, 50, 100]
num_tasks_list = [1000, 1000, 1000]
time_steps_per_epoch_list = [32 ,32, 32]

seeds = [1,2,3,4,5,6]

def compute_avg_loss_final_epoch(loss_array, num_tasks, num_epochs, time_steps_per_epoch):
  # Reshape loss_array into a 3D array: (tasks, epochs, time steps)
  loss_array = loss_array.reshape(num_tasks, num_epochs, time_steps_per_epoch)

  # Extract the losses for the final epoch (index num_epochs - 1) of each task
  final_epoch_losses = loss_array[:, num_epochs - 1, :]  # Shape: (num_tasks, time_steps_per_epoch)

  # Compute the average loss over the time steps for each task's final epoch
  average_losses = final_epoch_losses.mean(axis=1)  # Shape: (num_tasks,)

  return average_losses

for i in range(len(path_subroots)):
    path_subroot = path_subroots[i]
    epochs = epochs_list[i]
    num_tasks = num_tasks_list[i]
    time_steps_per_epoch = time_steps_per_epoch_list[i]

    avg_loss_fin_epoch_reset = np.zeros(num_tasks)
    avg_loss_fin_epoch_L2 = np.zeros(num_tasks)

    avg_loss_reset = np.zeros(num_tasks * epochs * time_steps_per_epoch)
    avg_loss_L2 = np.zeros(num_tasks * epochs * time_steps_per_epoch)

    for seed in seeds:
        path = path_root + path_subroot + f'/seed_{seed}/epochs_{epochs}/'
        with open(path + 'ART-L2_0.0001/loss_array.pkl', 'rb') as f:
            loss_array_reset = pickle.load(f)

        with open(path + 'L2_0.0001/loss_array.pkl', 'rb') as f:
            loss_array_L2 = pickle.load(f)
            print(path)
            print(np.mean(loss_array_L2))

        avg_loss_fin_epoch_reset += compute_avg_loss_final_epoch(loss_array_reset, num_tasks, epochs, time_steps_per_epoch)
        avg_loss_fin_epoch_L2 += compute_avg_loss_final_epoch(loss_array_L2, num_tasks, epochs, time_steps_per_epoch)

        avg_loss_reset += loss_array_reset
        avg_loss_L2 += loss_array_L2

    avg_loss_fin_epoch_reset = avg_loss_fin_epoch_reset / len(seeds)
    avg_loss_fin_epoch_L2 = avg_loss_fin_epoch_L2 / len(seeds)

    plt.plot(avg_loss_fin_epoch_L2, label = 'L2')
    plt.plot(avg_loss_fin_epoch_reset, label = 'Reset+L2')
    plt.title("Loss on Final Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Task")
    plt.show()
    plt.legend()
    plt.savefig(path_root + path_subroot+ "_avg_loss_final_epoch.png")
    plt.clf()

    # Compute the cumulative averages
    cumulative_avg_reset = np.cumsum(avg_loss_fin_epoch_reset) / np.arange(1, len(avg_loss_fin_epoch_reset) + 1)
    cumulative_avg_L2 = np.cumsum(avg_loss_fin_epoch_L2) / np.arange(1, len(avg_loss_fin_epoch_L2) + 1)

    # Plot the cumulative averages
    plt.plot(cumulative_avg_L2, label="L2")
    plt.plot(cumulative_avg_reset, label="Reset-L2")
    plt.title("Cumulative Average Loss on Final Epoch")
    plt.xlabel("Task")
    plt.ylabel("Cumulative Average Loss")
    plt.legend()
    plt.show()
    plt.savefig(path_root + path_subroot+ "_cumulative_avg_loss_final_epoch.png")
    plt.clf()

    # Compute the cumulative averages
    cumulative_avg_reset = np.cumsum(avg_loss_reset) / np.arange(1, len(avg_loss_reset) + 1)
    cumulative_avg_L2 = np.cumsum(avg_loss_L2) / np.arange(1, len(avg_loss_L2) + 1)

    # Plot the cumulative averages
    plt.plot(cumulative_avg_L2, label="L2")
    plt.plot(cumulative_avg_reset, label="Reset-L2")
    plt.title("Cumulative Average Loss over All Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Average Loss")
    plt.ylim(0,5)
    plt.legend()
    plt.show()
    plt.savefig(path_root + path_subroot+ "_cumulative_avg_loss.png")
    plt.clf()


