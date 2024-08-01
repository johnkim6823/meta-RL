import numpy as np

# Ensure reproducibility with a fixed seed
rng = np.random.RandomState(0)

# Define task generation functions
def gen_AR_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_HEALTH_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_HEAVY_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_INFOTAIN_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

def gen_new_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

# Dictionary of tasks
tasks = {
    "AUGMENTED_REALITY": gen_AR_task,
    "HEALTH_APP": gen_HEALTH_task,
    "HEAVY_COMP_APP": gen_HEAVY_task,
    "INFOTAINMENT_APP": gen_INFOTAIN_task
}

# Task conditions
task_conditions = {
    "AUGMENTED_REALITY": {
        "usage_percentage": 30,
        "prob_cloud_selection": 20,
        "poisson_interarrival": 2,
        "delay_sensitivity": 0.9,
        "active_period": 40,
        "idle_period": 20,
        "data_upload": 1500,
        "data_download": 25,
        "task_length": 9000,
        "required_core": 1,
        "vm_utilization_on_edge": 6,
        "vm_utilization_on_cloud": 0.6,
        "vm_utilization_on_mobile": 0
    },
    "HEALTH_APP": {
        "usage_percentage": 30,
        "prob_cloud_selection": 20,
        "poisson_interarrival": 3,
        "delay_sensitivity": 0.7,
        "active_period": 45,
        "idle_period": 90,
        "data_upload": 20,
        "data_download": 1250,
        "task_length": 3000,
        "required_core": 1,
        "vm_utilization_on_edge": 2,
        "vm_utilization_on_cloud": 0.2,
        "vm_utilization_on_mobile": 0    
    },
    "HEAVY_COMP_APP": {
        "usage_percentage": 20,
        "prob_cloud_selection": 40,
        "poisson_interarrival": 20,
        "delay_sensitivity": 0.1,
        "active_period": 60,
        "idle_period": 120,
        "data_upload": 2500,
        "data_download": 200,
        "task_length": 45000,
        "required_core": 1,
        "vm_utilization_on_edge": 30,
        "vm_utilization_on_cloud": 3,
        "vm_utilization_on_mobile": 0
    },
    "INFOTAINMENT_APP": {
        "usage_percentage": 30,
        "prob_cloud_selection": 10,
        "poisson_interarrival": 7,
        "delay_sensitivity": 0.3,
        "active_period": 30,
        "idle_period": 45,
        "data_upload": 25,
        "data_download": 1000,
        "task_length": 15000,
        "required_core": 1,
        "vm_utilization_on_edge": 10,
        "vm_utilization_on_cloud": 1,
        "vm_utilization_on_mobile": 0
    }
}
