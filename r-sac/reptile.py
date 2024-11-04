 # reptile.py
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def reptile(meta_model, tasks, num_inner_steps=5, inner_lr=0.01, meta_lr=0.001, print_interval=2, device='cpu'):
    meta_model = meta_model.to(device)
    meta_loss = 0.0
    
    print(f"Starting Reptile meta-learning on device: {device}")
    print(f"Number of tasks: {len(tasks)}, Inner steps per task: {num_inner_steps}")
    print(f"Inner learning rate: {inner_lr}, Meta learning rate: {meta_lr}")
    
    for task_idx, task in enumerate(tasks):
        print(f"\n--- Processing Task {task_idx + 1}/{len(tasks)} ---")
        
        # Initialize task model from meta_model for inner loop updates
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
            
            # Print loss for each inner step
            print(f"  Task {task_idx + 1} - Step {step + 1}/{num_inner_steps} - Loss: {loss.item():.4f}")
        
        # Meta-update of the meta model after inner updates
        print(f"Updating meta-model parameters after Task {task_idx + 1} inner loop.")
        for meta_param, task_param in zip(meta_model.parameters(), task_model.parameters()):
            meta_param.data = meta_param.data.to(device)
            task_param.data = task_param.data.to(device)
            meta_param.data += meta_lr * (task_param.data - meta_param.data)
        
        meta_loss += loss.item()
        
        # Print final loss after completing each task
        print(f"Task {task_idx + 1}/{len(tasks)} - Inner Loop Final Loss: {loss.item():.4f}")
    
    # Average meta-loss after all tasks
    avg_meta_loss = meta_loss / len(tasks)
    print(f"\n=== Average Meta Loss after all tasks: {avg_meta_loss:.4f} ===")
    return meta_model
