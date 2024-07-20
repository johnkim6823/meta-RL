import numpy as np
import random

# Ensure reproducibility with a fixed seed
rng = np.random.RandomState(0)

# Define task generation functions
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

# Define a function to generate tasks with conditions
def gen_task(task_conditions):
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    task_conditions["function"] = f_randomsine
    return task_conditions

# Dictionary of task generation functions
task_generators = {
    "COCO": gen_coco_task,
    "Google Speech Commands": gen_speech_commands_task,
    "Human Activity Recognition": gen_human_activity_task,
    "Urban Sound": gen_urban_sound_task,
    "CIFAR": gen_cifar_task,
    "New Task": gen_new_task
}

# Task conditions (provided)
task_conditions = {
    "COCO": {
        "usage_percentage": 20,
        "prob_cloud_selection": 30,
        "poisson_interarrival": 5,
        "delay_sensitivity": 1,
        "active_period": 60,
        "idle_period": 20,
        "data_upload": 1500,
        "data_download": 250,
        "task_length": 20000,
        "required_core": 4,
        "vm_utilization_on_edge": 8,
        "vm_utilization_on_cloud": 0.8,
        "vm_utilization_on_mobile": 20
    },
    "Google Speech Commands": {
        "usage_percentage": 20,
        "prob_cloud_selection": 20,
        "poisson_interarrival": 2,
        "delay_sensitivity": 1,
        "active_period": 45,
        "idle_period": 15,
        "data_upload": 200,
        "data_download": 1250,
        "task_length": 10000,
        "required_core": 2,
        "vm_utilization_on_edge": 4,
        "vm_utilization_on_cloud": 0.4,
        "vm_utilization_on_mobile": 10
    },
    "Human Activity Recognition": {
        "usage_percentage": 10,
        "prob_cloud_selection": 10,
        "poisson_interarrival": 3,
        "delay_sensitivity": 1,
        "active_period": 30,
        "idle_period": 20,
        "data_upload": 5,
        "data_download": 10,
        "task_length": 8000,
        "required_core": 2,
        "vm_utilization_on_edge": 2,
        "vm_utilization_on_cloud": 0.2,
        "vm_utilization_on_mobile": 5
    },
    "Urban Sound": {
        "usage_percentage": 25,
        "prob_cloud_selection": 25,
        "poisson_interarrival": 4,
        "delay_sensitivity": 1,
        "active_period": 50,
        "idle_period": 30,
        "data_upload": 100,
        "data_download": 500,
        "task_length": 15000,
        "required_core": 3,
        "vm_utilization_on_edge": 10,
        "vm_utilization_on_cloud": 1,
        "vm_utilization_on_mobile": 25
    },
    "CIFAR": {
        "usage_percentage": 25,
        "prob_cloud_selection": 15,
        "poisson_interarrival": 6,
        "delay_sensitivity": 1,
        "active_period": 40,
        "idle_period": 25,
        "data_upload": 20,
        "data_download": 100,
        "task_length": 12000,
        "required_core": 2,
        "vm_utilization_on_edge": 3.5,
        "vm_utilization_on_cloud": 0.35,
        "vm_utilization_on_mobile": 7
    },
}

# Generate task distribution
def generate_task_distribution(task_conditions, num_tasks=100):
    task_distribution = []
    for task_name, gen_task_func in task_generators.items():
        for _ in range(int(num_tasks * task_conditions[task_name]["usage_percentage"] / 100)):
            task = gen_task(task_conditions[task_name].copy())
            task_distribution.append(task)
    random.shuffle(task_distribution)
    return task_distribution

# Generate a distribution of 100 tasks
task_distribution = generate_task_distribution(task_conditions, num_tasks=100)

# Print some examples
for i, task in enumerate(task_distribution[:5]):
    print(f"Task {i+1}:")
    print(task)
    print()
