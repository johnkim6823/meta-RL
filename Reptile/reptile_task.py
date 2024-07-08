import os
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

# Ensure the directories exist
output_dir = "output_reptile_task"
training_dir = os.path.join(output_dir, "training")
test_dir = os.path.join(output_dir, "test")
os.makedirs(training_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

seed = 0
plot = True
innerstepsize = 0.02 # stepsize in inner SGD
innerepochs = 1 # number of epochs of each inner SGD
outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
niterations = 20000 # number of outer updates; each iteration we sample one task and update on it

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:, None] # All of the x points
ntrain = 10 # Size of training minibatches

def gen_coco_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_speech_commands_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_human_activity_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_urban_sound_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_cifar_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_new_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

tasks = {
    "COCO": gen_coco_task,
    "Google Speech Commands": gen_speech_commands_task,
    "Human Activity Recognition": gen_human_activity_task,
    "Urban Sound": gen_urban_sound_task,
    "CIFAR": gen_cifar_task,
}

task_conditions = {
    "COCO": {
        "innerstepsize": 0.02,
        "innerepochs": 1,
        "outerstepsize0": 0.1,
        "niterations": 30000,
        "data_upload": 100,
        "data_download": 500,
        "task_length": 20000,
        "required_core": 4
    },
    "Google Speech Commands": {
        "innerstepsize": 0.01,
        "innerepochs": 1,
        "outerstepsize0": 0.05,
        "niterations": 20000,
        "data_upload": 1,
        "data_download": 5,
        "task_length": 10000,
        "required_core": 2
    },
    "Human Activity Recognition": {
        "innerstepsize": 0.015,
        "innerepochs": 1,
        "outerstepsize0": 0.08,
        "niterations": 25000,
        "data_upload": 0.5,
        "data_download": 1,
        "task_length": 8000,
        "required_core": 2
    },
    "Urban Sound": {
        "innerstepsize": 0.02,
        "innerepochs": 1,
        "outerstepsize0": 0.1,
        "niterations": 30000,
        "data_upload": 10,
        "data_download": 50,
        "task_length": 15000,
        "required_core": 3
    },
    "CIFAR": {
        "innerstepsize": 0.018,
        "innerepochs": 1,
        "outerstepsize0": 0.09,
        "niterations": 28000,
        "data_upload": 2,
        "data_download": 10,
        "task_length": 12000,
        "required_core": 2
    },
}

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
f_plot = gen_coco_task()  # 예시로 COCO 작업을 고정
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]
xtest_plot = np.linspace(-5, 5, 100)[:, None]

# List to store loss values
loss_history = []

# Reptile training loop
for iteration in range(niterations):
    # 애플리케이션 무작위 선택
    task_name = rng.choice(list(tasks.keys()))
    gen_task = tasks[task_name]
    conditions = task_conditions[task_name]

    # Update innerstepsize, outerstepsize0, etc. based on the chosen task
    innerstepsize = conditions["innerstepsize"]
    outerstepsize0 = conditions["outerstepsize0"]

    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    y_all = f(x_all)
    # Do SGD on this task
    inds = rng.permutation(len(x_all))
    for _ in range(conditions["innerepochs"]):
        for start in range(0, len(x_all), ntrain):
            mbinds = inds[start:start+ntrain]
            train_on_batch(x_all[mbinds], y_all[mbinds])
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / conditions["niterations"]) # linear schedule
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
        for name in weights_before})

    # Periodically plot the results on a particular task and minibatch
    if plot and iteration == 0 or (iteration + 1) % 1000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for inneriter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter + 1) % 8 == 0:
                frac = (inneriter + 1) / 32
                plt.plot(x_all, predict(x_all), label="pred after %i"%(inneriter + 1), color=(frac, 0, 1 - frac))
        plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.ylim(-4, 4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model Predictions during Training')
        plt.legend(loc="lower right")
        plt.pause(0.01)
        
        # Save the plot every 1000 iterations
        if (iteration + 1) % 1000 == 0:
            plt.savefig(os.path.join(training_dir, f"plot_iteration_{iteration + 1}.png"))
        
        model.load_state_dict(weights_before) # restore from snapshot
        print(f"-----------------------------")
        print(f"iteration               {iteration + 1}")
        print(f"loss on plotted curve   {lossval:.3f}") # would be better to average loss over a set of examples, but this is optimized for brevity

    # Save loss value
    loss_history.append(lossval)

# Plot the loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.savefig(os.path.join(training_dir, "loss_history.png"))
plt.close()

#-------------------TESTING SECTION-------------------
print("Testing the model on new tasks.......")

# Function to test the model on new tasks
def test_model(gen_task, num_iterations=5000, test_task_name="New Task"):
    f_test = gen_task()
    x_test = np.linspace(-5, 5, 50)[:, None]
    y_test = f_test(x_test)
    
    # List to store test losses
    test_losses = []

    # Initial predictions
    plt.figure()
    plt.plot(x_test, y_test, label="True Function", color='g')
    plt.plot(x_test, predict(x_test), label="Initial Prediction", color='r')
    plt.pause(0.01)
    
    best_iteration = 0
    best_loss = float('inf')
    best_prediction = None
    
    for i in range(num_iterations):
        train_on_batch(x_test, y_test)
        if (i + 1) % (num_iterations // 5) == 0 or i == num_iterations - 1:
            current_prediction = predict(x_test)
            plt.plot(x_test, current_prediction, label=f"Prediction after {i + 1} iterations")
            plt.pause(0.01)
            test_loss = np.square(current_prediction - y_test).mean()
            test_losses.append(test_loss)
            print(f"Iteration {i + 1}, Test Loss: {test_loss:.3f}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_iteration = i + 1
                best_prediction = current_prediction
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Model Predictions on {test_task_name}')
    plt.legend(loc="lower right")
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
    plt.savefig(os.path.join(test_dir, f"best_prediction_{test_task_name.replace(' ', '_')}.png"))
    plt.close()
    
    return test_losses

# Test the model on new tasks
new_task_losses = test_model(gen_new_task, test_task_name="New Task Example")

# Plot the test loss history
plt.figure()
plt.plot(new_task_losses)
plt.xlabel('Iteration')
plt.ylabel('Test Loss')
plt.title('Test Loss over Iterations for New Task')
plt.savefig(os.path.join(test_dir, "test_loss_history.png"))
plt.close()

print("Training and Testing Completed!")
print(f"Results saved in {output_dir} directory.")
