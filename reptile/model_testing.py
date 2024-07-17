import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import autograd as ag
from matplotlib.ticker import FuncFormatter
from copy import deepcopy
from tasks import tasks  # Import the tasks dictionary

def totorch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y, model, innerstepsize):
    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()
    for param in model.parameters():
        param.data -= innerstepsize * param.grad.data

def predict(x, model):
    x = totorch(x)
    return model(x).data.numpy()

def test_model(model, gen_task, test_dir, num_iterations=1000, test_task_name="New Task", patience=10, min_loss_diff=0.01):
    task_frequency = {task_name: 0 for task_name in tasks.keys()}
    f_test = gen_task()
    x_test = np.linspace(-5, 5, 50)[:, None]
    y_test = f_test(x_test)
    
    task_name = "Unknown Task"
    for name, gen_func in tasks.items():
        if gen_func == gen_task:
            task_name = name
            task_frequency[task_name] += 1
            break

    # Print task frequency distribution at the beginning
    print(f"Test Task Frequency Distribution:")
    for task, freq in task_frequency.items():
        print(f"{task}: {freq} times")
    print(f"-----------------------------")

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
    plt.plot(x_test, predict(x_test, model), label="Initial Prediction", color='r')
    
    best_iteration = 0
    best_loss = float('inf')
    best_prediction = None
    no_improvement_count = 0
    
    for i in range(num_iterations):
        train_on_batch(x_test, y_test, model, innerstepsize=0.05)
        current_prediction = predict(x_test, model)
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
    plt.plot(x_test, best_prediction, label=f'Best Prediction after {best_iteration} iterations', color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Best Model Prediction on {test_task_name}')
    plt.legend(loc="lower right")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    plt.savefig(os.path.join(test_dir, f"best_prediction_{test_task_name.replace(' ', '_')}.png"))
    plt.close()
    
    return test_losses
