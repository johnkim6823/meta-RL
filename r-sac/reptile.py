import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def reptile(meta_model, tasks, num_inner_steps=5, inner_lr=0.01, meta_lr=0.001, print_interval=2, device='cpu'):
    meta_model = meta_model.to(device)
    meta_loss = 0.0
    
    for task_idx, task in enumerate(tasks):
        if (task_idx + 1) % print_interval == 0 or task_idx == len(tasks) - 1:
            print(f"\n--- Processing Task {task_idx + 1}/{len(tasks)} ---")
        
        task_model = deepcopy(meta_model).to(device)
        inner_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)
        
        for step in range(num_inner_steps):
            inputs, targets = task
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = task_model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        for meta_param, task_param in zip(meta_model.parameters(), task_model.parameters()):
            meta_param.data = meta_param.data.to(device)
            task_param.data = task_param.data.to(device)
            meta_param.data += meta_lr * (task_param.data - meta_param.data)
        
        meta_loss += loss.item()
        
        if (task_idx + 1) % print_interval == 0 or task_idx == len(tasks) - 1:
            print(f"Task {task_idx + 1}/{len(tasks)} - Inner Loop Final Loss: {loss.item():.4f}")
    
    avg_meta_loss = meta_loss / len(tasks)
    print(f"\n=== Average Meta Loss after all tasks: {avg_meta_loss:.4f} ===")
    return meta_model