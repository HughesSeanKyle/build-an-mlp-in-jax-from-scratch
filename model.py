"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed: int) -> jax.Array:
    """Wraps an integer seed into a standard JAX PRNG key array."""
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax

def split_prng_key(key: jax.Array, num: int) -> jax.Array:
    """Splits a parent JAX PRNG key into num independent subkeys."""
    return jax.random.split(key, num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key: jax.Array, shape: tuple) -> jax.Array:
    """Draws a standard-normal sample of the requested shape using a PRNG key."""
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key: jax.Array, batch_size: int, num_features: int) -> jax.Array:
    """Draws a batch of standard-normal feature vectors using sample_normal_matrix."""
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
def assign_class_labels(inputs: jax.Array, num_classes: int) -> jax.Array:
    """Turns a feature matrix into discrete integer labels via an argmax rule."""
    return jnp.argmax(inputs[:, :num_classes], axis=1).astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels: jax.Array, num_classes: int) -> jax.Array:
    """Converts a 1-D array of integer class labels into a 2-D one-hot matrix."""
    return jax.nn.one_hot(labels, num_classes)

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key: jax.Array, in_dim: int, out_dim: int, scale=0.1) -> dict:
    """
    Builds initialized parameter weights and zero biases for one linear layer.
    Uses the exact input key directly on sample_normal_matrix to match test references.
    """
    # Pass the input key directly without splitting first
    W = sample_normal_matrix(key, (in_dim, out_dim)) * scale
    b = jnp.zeros((out_dim,))
    return {'W': W, 'b': b}

# Step 8 - init_mlp_params
def init_mlp_params(key: jax.Array, layer_sizes: list, scale=0.1) -> list:
    """Builds a full list of initialized layer parameter dicts for an MLP."""
    num_layers = len(layer_sizes) - 1
    keys = split_prng_key(key, num_layers)
    
    params = []
    for i in range(num_layers):
        layer_param = init_linear_layer(keys[i], layer_sizes[i], layer_sizes[i+1], scale=scale)
        params.append(layer_param)
    return params

# Step 9 - linear_forward
def linear_forward(x: jax.Array, layer_params: dict) -> jax.Array:
    """Performs a single affine transformation (x @ W + b) of a batch."""
    return x @ layer_params['W'] + layer_params['b']

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x: jax.Array) -> jax.Array:
    """Applies the Rectified Linear Unit activation function elementwise."""
    return jnp.maximum(0, x)

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits: jax.Array) -> jax.Array:
    """Converts a logits array into standard probabilities along the last axis."""
    return jax.nn.softmax(logits, axis=-1)

# Step 12 - mlp_forward
def mlp_forward(params: list, x: jax.Array) -> jax.Array:
    """Runs an input batch through the MLP layers up to raw output logits."""
    activation = x
    # Forward through all hidden layers with ReLU
    for layer_param in params[:-1]:
        linear_out = linear_forward(activation, layer_param)
        activation = relu_activation(linear_out)
    
    # Final layer only uses the affine transformation (no activation)
    return linear_forward(activation, params[-1])

# Step 13 - log_softmax_logits
def log_softmax_logits(logits: jax.Array) -> jax.Array:
    """Returns a stable log of the softmax distribution along the last axis."""
    return jax.nn.log_softmax(logits, axis=-1)

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits: jax.Array, one_hot_targets: jax.Array) -> jax.Array:
    """Computes the mean cross-entropy between predicted logits and targets."""
    log_probs = log_softmax_logits(logits)
    # Categorical cross entropy: mean of (-target * log_prob)
    return -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Computes the fraction of correctly classified examples as a scalar float."""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# Step 16 - loss_fn_of_params
import jax
import jax.numpy as jnp

def loss_fn_of_params(params: list, x: jax.Array, one_hot_targets: jax.Array) -> jax.Array:
    """A target-first function wrapper evaluating scalar loss given parameters."""
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)

# Step 17 - compute_param_grads
import jax
import jax.numpy as jnp

def compute_param_grads(params: list, x: jax.Array, one_hot_targets: jax.Array) -> list:
    """Computes analytical gradient pytrees of the loss with respect to parameters."""
    grad_fn = jax.grad(loss_fn_of_params, argnums=0)
    return grad_fn(params, x, one_hot_targets)

# Step 18 - sgd_update_params
import jax
import jax.numpy as jnp

def sgd_update_params(params: list, grads: list, learning_rate: float) -> list:
    """Applies a single vanilla functional SGD step across all parameters."""
    return [
        {
            'W': layer_p['W'] - learning_rate * layer_g['W'],
            'b': layer_p['b'] - learning_rate * layer_g['b']
        }
        for layer_p, layer_g in zip(params, grads)
    ]

# Step 19 - training_step
import jax
import jax.numpy as jnp

def training_step(params: list, x: jax.Array, one_hot_targets: jax.Array, learning_rate: float) -> tuple:
    """Runs a unitary training pass, returning updated parameters and old loss."""
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    new_params = sgd_update_params(params, grads, learning_rate)
    return new_params, loss

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

