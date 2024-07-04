import tensorflow as tf
import numpy as np

def sample_points(k):
    num_points = 100
    
    # amplitude
    amplitude = np.random.uniform(low=0.1, high=5.0)
    
    # phase
    phase = np.random.uniform(low=0, high=np.pi)

    x = np.linspace(-5, 5, num_points)

    # y = a*sin(x+b)
    y = amplitude * np.sin(x + phase)
    
    # sample k data points
    sample = np.random.choice(np.arange(num_points), size=k)
    
    return (x[sample], y[sample])

x, y = sample_points(10)
print(x)
print(y)

num_hidden = 64
num_classes = 1
num_feature = 1

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_hidden, activation='tanh', input_shape=(num_feature,)),
    tf.keras.layers.Dense(num_classes, activation='tanh')
])

# Define the loss function and optimizer
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-2)

# number of epochs i.e training iterations
num_epochs = 100

# number of samples i.e number of shots
num_samples = 50  

# number of tasks
num_tasks = 2

# number of times we want to perform optimization
num_iterations = 10

# mini batch size
mini_batch = 10  

# Training loop
for e in range(num_epochs):
    for task in range(num_tasks):
        # Get the initial parameters of the model
        old_weights = model.get_weights()

        # Sample x and y
        x_sample, y_sample = sample_points(num_samples)
        x_sample = x_sample.reshape(-1, 1)
        y_sample = y_sample.reshape(-1, 1)

        # For some k number of iterations perform optimization on the task
        for k in range(num_iterations):
            # Get the minibatch x and y
            for i in range(0, num_samples, mini_batch):
                # Sample mini batch of examples 
                x_minibatch = x_sample[i:i+mini_batch]
                y_minibatch = y_sample[i:i+mini_batch]

                with tf.GradientTape() as tape:
                    y_pred = model(x_minibatch, training=True)
                    loss = loss_function(y_minibatch, y_pred)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Get the updated model parameters after several iterations of optimization
        new_weights = model.get_weights()

        # Now we perform meta update
        epsilon = 0.1
        updated_weights = [old_w + epsilon * (new_w - old_w) for old_w, new_w in zip(old_weights, new_weights)]

        # Update the model parameter with new parameters
        model.set_weights(updated_weights)

    if e % 10 == 0:
        y_pred = model(x_sample, training=False)
        loss = loss_function(y_sample, y_pred)
        print("Epoch {}: Loss {}\n".format(e, loss))
        print('Updated Model Parameter Theta\n')
        print('Sampling Next Batch of Tasks \n')
        print('---------------------------------\n')
