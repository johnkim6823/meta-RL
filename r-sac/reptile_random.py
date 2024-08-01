import os
import shutil
import time
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from copy import deepcopy
from tasks import *
from datetime import datetime

# Ensure the directories exist and clear them if they already exist
output_dir = "output_random"
training_dir = os.path.join(output_dir, "training")

# Function to clear and recreate directories
def reset_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

reset_directory(training_dir)

seed = 0
plot = True
innerstepsize = 0.02  # stepsize in inner SGD
innerepochs = 1  # number of epochs of each inner SGD
inneriter = 32  # number of inner SGD iterations
outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization
niterations = 10000  # number of outer updates; each iteration we sample one task and update on it

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:,None] # All of the x points
ntrain = 10 # Size of training minibatches

def gen_task():
    "Generate classification problem"
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x : np.sin(x + phase) * ampl
    return f_randomsine

# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

def totorch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()
    for param in model.parameters():
        param.data -= innerstepsize * param.grad.data

def predict(x):
    x = totorch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]

# List to store loss values
loss_history = []

# Dictionary to store task frequency
task_frequency = {task_name: 0 for task_name in tasks.keys()}

# Start measuring training time
start_training_time = time.time()

# Reptile training loop
for iteration in range(niterations):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    y_all = f(x_all)
    # Do SGD on this task
    inds = rng.permutation(len(x_all))
    for _ in range(innerepochs):
        for start in range(0, len(x_all), ntrain):
            mbinds = inds[start:start+ntrain]
            train_on_batch(x_all[mbinds], y_all[mbinds])
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
        for name in weights_before})

    # Periodically plot the results on a particular task and minibatch
    if plot and iteration == 0 or (iteration + 1) % 1000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for inneriter_idx in range(inneriter):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter_idx + 1) % 8 == 0:
                frac = (inneriter_idx + 1) / inneriter
                plt.plot(x_all, predict(x_all), label="pred after %i"%(inneriter_idx + 1), color=(frac, 0, 1 - frac))
        plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.ylim(-4, 4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model Predictions during Training')
        plt.legend(loc="lower right")
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Save the plot every 1000 iterations
        if (iteration + 1) % 1000 == 0:
            plt.savefig(os.path.join(training_dir, f"plot_iteration_{iteration + 1}.png"))
        
        model.load_state_dict(weights_before) # restore from snapshot
        print(f"-----------------------------")
        print(f"iteration               {iteration + 1}")
        print(f"loss on plotted curve   {lossval:.3f}") # would be better to average loss over a set of examples, but this is optimized for brevity

    # Save loss value
    loss_history.append(lossval)

# End measuring training time
end_training_time = time.time()
training_time = end_training_time - start_training_time

# Print the final weights
final_weights = {name: param.data for name, param in model.named_parameters()}
for name, weight in final_weights.items():
    print(f"{name}: {weight}")

print(f"Training completed in {training_time:.2f} seconds")

# Plot the loss history with detailed view
plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations (Full View)')
plt.ylim(bottom=0)  # Set y-axis to start at 0 for better readability
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
plt.savefig(os.path.join(training_dir, "loss_history_full.png"))
plt.close()

# Plot the loss history with zoomed-in view
plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations (Zoomed View)')
plt.ylim(0, max(loss_history) * 0.1)  # Zoom into the first 10% of the maximum loss
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.4f}'))
plt.savefig(os.path.join(training_dir, "loss_history_zoomed.png"))
plt.close()


print(f"Training completed in {training_time:.2f} seconds")