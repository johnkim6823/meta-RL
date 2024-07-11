import os
import shutil
import time
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from copy import deepcopy
from tasks import tasks, task_conditions, gen_new_task

# Ensure the directories exist and clear them if they already exist
output_dir = "output_reptile_task"
training_dir = os.path.join(output_dir, "training")
test_dir = os.path.join(output_dir, "test")

# Function to clear and recreate directories
def reset_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

reset_directory(training_dir)
reset_directory(test_dir)

seed = 0
plot = True
innerstepsize = 0.05  # stepsize in inner SGD
innerepochs = 1  # number of epochs of each inner SGD
inneriter = 32  # number of inner SGD iterations
outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization
niterations = 10000  # number of outer updates; each iteration we sample one task and update on it

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:, None]  # All of the x points
ntrain = 10  # Size of training minibatches

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
f_plot = tasks["Urban Sound"]()  # Fixed task for consistent visualization
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]
xtest_plot = np.linspace(-5, 5, 100)[:, None]

# List to store loss values
loss_history = []

# Start measuring training time
start_training_time = time.time()

# Reptile training loop
for iteration in range(niterations):
    # Randomly select an application (task)
    task_name = rng.choice(list(tasks.keys()))
    gen_task = tasks[task_name]
    conditions = task_conditions[task_name]

    # Print training metadata
    if iteration == 0 or (iteration + 1) % 1000 == 0:
        print("-----------------------------")
        print(f"Iteration {iteration + 1}")
        print(f"Training on task: {task_name}")
        #print(f"Task conditions: {conditions}")


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
        #print(f"-----------------------------")
        #print(f"iteration               {iteration + 1}")
        print(f"loss on plotted curve   {lossval:.3f}") # would be better to average loss over a set of examples, but this is optimized for brevity

    # Save loss value
    loss_history.append(lossval)

# End measuring training time
end_training_time = time.time()
training_time = end_training_time - start_training_time

# Plot the loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.ylim(bottom=0)  # Set y-axis to start at 0 for better readability
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
plt.savefig(os.path.join(training_dir, "loss_history.png"))
plt.close()

#-------------------TESTING SECTION-------------------
print(f"-----------------------------")
print("Testing the model on new tasks.......")

# Function to test the model on new tasks with early stopping
def test_model(gen_task, num_iterations=1000, test_task_name="New Task", patience=10, min_loss_diff=0.01):
    f_test = gen_task()
    x_test = np.linspace(-5, 5, 50)[:, None]
    y_test = f_test(x_test)
    
    # List to store test losses
    test_losses = []

    # Print test metadata
    print(f"Testing on task: {test_task_name}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Patience: {patience}")
    print(f"Minimum loss difference for early stopping: {min_loss_diff}")
    print(f"-----------------------------")

    # Initial predictions
    plt.figure()
    plt.plot(x_test, y_test, label="True Function", color='g')
    plt.plot(x_test, predict(x_test), label="Initial Prediction", color='r')
    
    best_iteration = 0
    best_loss = float('inf')
    best_prediction = None
    no_improvement_count = 0
    
    for i in range(num_iterations):
        train_on_batch(x_test, y_test)
        current_prediction = predict(x_test)
        test_loss = np.square(current_prediction - y_test).mean()
        test_losses.append(test_loss)
        print(f"Iteration {i + 1}, Test Loss: {test_loss:.3f}")

        if (i + 1) % (num_iterations // 5) == 0 or i == num_iterations - 1:
            plt.plot(x_test, current_prediction, label=f"Prediction after {i + 1} iterations")

        if test_loss < best_loss:
            if best_loss - test_loss < min_loss_diff:
                print(f"Stopping early at iteration {i + 1} due to minimal loss improvement.")
                print(f"-----------------------------")
                break
            best_loss = test_loss
            best_iteration = i + 1
            best_prediction = current_prediction
            no_improvement_count = 0  # Reset the counter if improvement is found
        else:
            no_improvement_count += 1  # Increment the counter if no improvement
        
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {i + 1} with best iteration {best_iteration}")
            print(f"-----------------------------")
            break
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Model Predictions on {test_task_name}')
    plt.legend(loc="lower right")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    plt.savefig(os.path.join(test_dir, f"test_{test_task_name.replace(' ', '_')}.png"))
    plt.close()
    
    # Plot best prediction
    plt.figure()
    plt.plot(x_test, y_test, label="True Function", color='g')
    plt.plot(x_test, best_prediction, label=f"Best Prediction after {best_iteration} iterations", color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Best Model Prediction on {test_task_name}')
    plt.legend(loc="lower right")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    plt.savefig(os.path.join(test_dir, f"best_prediction_{test_task_name.replace(' ', '_')}.png"))
    plt.close()
    
    return test_losses

# Start measuring testing time
start_testing_time = time.time()

# Test the model on new tasks with early stopping
new_task_losses = test_model(gen_new_task, test_task_name="New Task Example", patience=10, min_loss_diff=0.001)

# End measuring testing time
end_testing_time = time.time()
testing_time = end_testing_time - start_testing_time

# Plot the test loss history with integer x-axis
plt.figure()
plt.plot(range(1, len(new_task_losses) + 1), new_task_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Test Loss over Iterations on New Environment')
plt.xticks(range(1, len(new_task_losses) + 1))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
plt.savefig(os.path.join(test_dir, "test_loss_history.png"))
plt.close()

print("Training and Testing Completed!")
print(f"Results saved in {output_dir} directory.")
print(f"Total Training Time: {training_time:.2f} seconds")
print(f"Total Testing Time: {testing_time:.2f} seconds")
print(f"-----------------------------")

# Plot the loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.ylim(bottom=0)  # Set y-axis to start at 0 for better readability
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
plt.savefig(os.path.join(training_dir, "loss_history.png"))
plt.close()
