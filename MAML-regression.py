import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import os

# Get input from user
print("Meta-Learning with MAML for Regression")
print("-" * 50)
num_epochs_input = int(input("Enter the number of epochs: "))
num_epochs = num_epochs_input + 1

batch_size_options = [64, 128, 256]
learning_rate_options = [1e-2, 1e-3, 1e-4]

def select_option(options, option_name):
    print(f"Select {option_name}:")
    for i, option in enumerate(options, 1):
        print(f"{i}: {option}")
    choice = input(f"Enter the number corresponding to your choice or 'a' for all options: ").strip().lower()
    if choice == 'a':
        return options
    else:
        return [options[int(choice) - 1]]

batch_size_choices = select_option(batch_size_options, "batch size")
learning_rate_choices = select_option(learning_rate_options, "learning rate")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-" * 50)
print(device)
print("-" * 50)

# Base directory for saving plots
base_dir = "output_MAML-regression"
os.makedirs(base_dir, exist_ok=True)

class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len

class SinusoidalFunction:
    def __init__(self, x_range=5, k=5, num_tasks=4):
        self.x_range = x_range
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.k = k
        self.num_tasks = num_tasks

    def meta_train_data(self, batch_size):
        x_points = 2 * self.x_range * (torch.rand((self.num_tasks, 2 * self.k)) - 0.5)
        y_points = torch.tensor([], dtype=torch.float)

        for x in x_points:
            a = 4 * (torch.rand(1) + 0.1)
            b = self.pi * torch.rand(1)
            y = a * torch.sin(x.view(1, -1) + b)
            y_points = torch.cat((y_points, y), 0)

        taskset = TensorData(x_points, y_points)
        trainloader = torch.utils.data.DataLoader(taskset, batch_size=batch_size)
        return trainloader

    def meta_eval_data(self, k):
        x_points = 2 * self.x_range * (torch.rand(2 * k) - 0.5)
        a = 4 * (torch.rand(1) + 0.1)
        b = self.pi * torch.rand(1)
        y_points = a * torch.sin(x_points + b)
        sup_x = x_points[:k]
        sup_y = y_points[:k]
        que_x = x_points[k:]
        que_y = y_points[k:]
        x = torch.linspace(-self.x_range, self.x_range, 200)
        y = a * torch.sin(x + b)
        return sup_x, sup_y, que_x, que_y, x, y
    
k = 5
num_tasks = 2000
sine = SinusoidalFunction(k=k, num_tasks=num_tasks)

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def parameterised(self, x, weights):
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x    
    
class MAML:

    def __init__(self, trainloader, k, alpha, beta=1e-3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        self.k = k
        self.model = Regressor().to(device)
        self.weights = list(self.model.parameters())
        self.trainloader = trainloader
        self.beta = beta
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.weights, lr=alpha)

    def inner_loop(self, data):
        temp_weights = [w.clone() for w in self.weights]
        
        inputs, values = data[0].to(device), data[1].to(device)
        support_x = inputs[:, :self.k].T.reshape(-1, 1)  # Reshape support_x
        support_y = values[:, :self.k].T.reshape(-1, 1)  # Reshape support_y
        query_x = inputs[:, self.k:].T.reshape(-1, 1)  # Reshape query_x
        query_y = values[:, self.k:].T.reshape(-1, 1)  # Reshape query_y

        outputs = self.model.parameterised(support_x, temp_weights)
        loss = self.criterion(outputs, support_y)
        grad = torch.autograd.grad(loss, temp_weights)
        tmp = [w - self.beta * g for w, g in zip(temp_weights, grad)]
            
        outputs = self.model.parameterised(query_x, tmp)
        inner_loss = self.criterion(outputs, query_y)

        return inner_loss

    def meta_train(self, num_epochs):
        n = len(self.trainloader)
        loss_list = []

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outer_loss = 0

            for data in self.trainloader:
                outer_loss += self.inner_loop(data)
                
            avg_loss = outer_loss / n
            avg_loss.backward()
            self.optimizer.step()
            ll = avg_loss.item()
            loss_list.append(ll)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {ll}")

        print("-" * 50)
        return loss_list

def run_experiment(batch_size, learning_rate):
    trainloader = sine.meta_train_data(batch_size)

    # Dynamic setting value
    setting = f"epochs_{num_epochs}_batch_{batch_size}_learningRate_{learning_rate}"
    setting_dir = os.path.join(base_dir, setting)
    os.makedirs(setting_dir, exist_ok=True)

    # Initial SinusoidalFunction class visualization
    plt.figure(figsize=(10, 5))
    for i in range(5):
        _, _, _, _, x, y = sine.meta_eval_data(5)
        plt.plot(x, y)
    plt.xlabel("Angle (radians)")
    plt.ylabel("Sine value")
    plt.title("Initial Sinusoidal Function Samples")
    initial_plot_path = os.path.join(setting_dir, "initial_sinusoidal_function_samples.png")
    plt.savefig(initial_plot_path)
    print(f"Saved plot: {initial_plot_path}")
    plt.close()

    maml = MAML(trainloader, k=k, alpha=learning_rate)
    loss = maml.meta_train(num_epochs)

    # Loss visualization
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.title("MAML Training Loss Over Epochs")
    loss_plot_path = os.path.join(setting_dir, f"maml_training_loss_epochs_{num_epochs_input}_batch_{batch_size}_learningRate_{learning_rate}.png")
    plt.savefig(loss_plot_path)
    print(f"Saved plot: {loss_plot_path}")
    plt.close()

    # Inference visualization
    def inference(sup_x, sup_y, x, y, model, title, filename):
        with torch.no_grad():
            pred = model(x.view(-1, 1).to(device))
            plt.figure(figsize=(10, 5))
            plt.plot(x.cpu().detach(), pred.cpu().detach(), '-b')
            plt.plot(sup_x.cpu().detach(), sup_y.cpu().detach(), '.g')
            plt.plot(x, y, '--r')
            plt.legend(['Prediction', 'Support Points', 'True Function'])
            plt.xlabel("Angle (radians)")
            plt.ylabel("Sine value")
            plt.title(title)
            plt.savefig(filename)
            print(f"Saved plot: {filename}")
            plt.close()

    sup_x, sup_y, _, _, x, y = sine.meta_eval_data(5)
    plt.plot(x, y)
    plt.plot(sup_x, sup_y, '.')
    plt.xlabel("Angle (radians)")
    plt.ylabel("Sine value")
    plt.title("Evaluation Data for Sinusoidal Function")
    evaluation_plot_path = os.path.join(setting_dir, f"evaluation_data_sinusoidal_function_epochs_{num_epochs_input}_batch_{batch_size}_learningRate_{learning_rate}.png")
    plt.savefig(evaluation_plot_path)
    print(f"Saved plot: {evaluation_plot_path}")
    plt.close()

    # MAML model inference
    pre = maml.model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(pre.parameters(), lr=learning_rate)
    for i in range(num_epochs):
        optimizer.zero_grad()
        outputs = pre(sup_x.view(-1, 1).to(device))
        loss = criterion(outputs, sup_y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        if i == 0:
            inference(sup_x, sup_y, x, y, pre, f"MAML Model Inference at Epoch 0", os.path.join(setting_dir, f"maml_inference_epoch_0_epochs_{num_epochs_input}_batch_{batch_size}_learningRate_{learning_rate}.png"))
        elif i == num_epochs - 1:
            inference(sup_x, sup_y, x, y, pre, f"MAML Model Inference at Epoch {num_epochs_input}", os.path.join(setting_dir, f"maml_inference_epoch_{num_epochs_input}_epochs_{num_epochs_input}_batch_{batch_size}_learningRate_{learning_rate}.png"))

    # Baseline model inference
    net = Regressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for i in range(num_epochs):
        optimizer.zero_grad()
        outputs = net(sup_x.view(-1, 1).to(device))
        loss = criterion(outputs, sup_y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        if i == 0:
            inference(sup_x, sup_y, x, y, net, f"Baseline Model Inference at Epoch 0", os.path.join(setting_dir, f"baseline_inference_epoch_0_epochs_{num_epochs_input}_batch_{batch_size}_learningRate_{learning_rate}.png"))
        elif i == num_epochs - 1:
            inference(sup_x, sup_y, x, y, net, f"Baseline Model Inference at Epoch {num_epochs_input}", os.path.join(setting_dir, f"baseline_inference_epoch_{num_epochs_input}_epochs_{num_epochs_input}_batch_{batch_size}_learningRate_{learning_rate}.png"))

for batch_size in batch_size_choices:
    for learning_rate in learning_rate_choices:
        run_experiment(batch_size, learning_rate)
