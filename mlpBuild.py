import jax
import jax.numpy as jnp
import numpy as np
 
 
 
def make_prng_key(seed):
    # TODO: wrap a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))
    return jax.random.PRNGKey(seed)
    pass

def split_prng_key(key, num):
    # TODO: split `key` into `num` independent subkeys and return them as a (num, 2) array.
    return jax.random.split(key,num)
    pass
 
 
def sample_normal_matrix(key, shape):
    return jax.random.normal(key, shape=shape)
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    pass
 
 
def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # TODO: draw a batch of random input feature vectors from the PRNG key
    return sample_normal_matrix(key, (batch_size , num_features))
    pass
 
def assign_class_labels(inputs, num_classes):
    sliced_inputs = inputs[:, :num_classes]
    # TODO: return an int32 label per row using the first num_classes feature columns.
    return jnp.argmax(sliced_inputs ,axis=1).astype("int32")
    pass
 
def one_hot_encode_labels(labels, num_classes):
    return jax.nn.one_hot(labels, num_classes , dtype = jnp.float32  )
    # TODO: Convert a 1-D array of integer class indices into a 2-D one-hot matrix of shape (batch, num_classes).
    pass
 
 
def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    # TODO: sample W from a scaled normal and set b to zeros, return as a dict.
    W = sample_normal_matrix(key, (in_dim,out_dim))* scale
    b = jnp.zeros((out_dim))
    return {"W" : W , "b" : b}
    pass
 
def init_mlp_params(key, layer_sizes, scale=0.1):
    # TODO: build a list of per-layer parameter dicts from adjacent layer sizes.
    subkeys = jax.random.split(key, len(layer_sizes) - 1)
    return [
        init_linear_layer(subkey, in_dim, out_dim, scale=scale)
        for subkey, in_dim, out_dim in zip(subkeys, layer_sizes[:-1], layer_sizes[1:])
    ]
    pass
 
def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    
    return jnp.matmul(x, layer_params["W"] )+ layer_params["b"]
 
 
 
def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # TODO: return an array of the same shape with negatives replaced by zero.
    return jax.nn.relu(x)
    pass
 
 
def softmax_probabilities(logits):
    # TODO: convert logits into a numerically stable softmax along the last axis
    return jax.nn.softmax(logits)
    pass
 
def mlp_forward(params, x):
    for layer_params in params[:-1]:
        x = linear_forward(x, layer_params)
        x = relu_activation(x)
 
    return linear_forward(x, params[-1])
 
def log_softmax_logits(logits):
    # TODO: return the numerically stable log-softmax of logits along the last axis.
    return jax.nn.log_softmax(logits, axis=-1)
 
    pass
 
def cross_entropy_loss(logits, one_hot_targets):
    log_probs = log_softmax_logits(logits)
    loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    return jnp.mean(loss)
 
def classification_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return accuracy
 
 
def loss_fn_of_params(params, x, one_hot_targets):
    logits = mlp_forward(params , x)
    return cross_entropy_loss(logits , one_hot_targets)
    # TODO: return scalar cross-entropy loss as a function of params, ready for jax.grad
    pass
 
 
def compute_param_grads(params, x, one_hot_targets):
    # TODO: return grad of loss_fn_of_params w.r.t. params using jax.grad
    return jax.grad(loss_fn_of_params ,argnums=0)(params,x,one_hot_targets)
 
 
def sgd_update_params(params, grads, learning_rate):
    # TODO: apply one SGD step to every parameter using its gradient and a learning rate
 
 
    return [
        {
            'W': p['W'] - learning_rate * g['W'],
            'b': p['b'] - learning_rate * g['b']
        }
        for p, g in zip(params, grads)
    ]
 
 
def training_step(params, x, one_hot_targets, learning_rate):
    # TODO: compute current loss + grads via the upstream helpers, then SGD-update params.
    loss = loss_fn_of_params(params , x, one_hot_targets)
 
    grads = compute_param_grads(params , x, one_hot_targets)
 
    new_params = sgd_update_params(params , grads , learning_rate)
    return new_params , loss
 
    pass
 
def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Run num_epochs full-batch SGD updates and return the final params."""
    # TODO: run num_epochs full-batch SGD updates via training_step and return final params
    for i in range (num_epochs):
        params, i = training_step(params , x, one_hot_targets , learning_rate)
    return params
    pass
 
def predict_classes(params, x):
    # TODO: run mlp_forward on x and return the argmax class index per row
 
    return jnp.argmax(mlp_forward(params,x), axis=-1)
    pass
 
"""End-to-end demo: train a small MLP in JAX from scratch on synthetic data."""
 
 
 
if __name__ == "__main__":
    np.random.seed(0)
 
    # --- Config ---
    seed = 0
    batch_size = 32
    num_features = 8
    num_classes = 4
    hidden_size = 16
    learning_rate = 0.1
    num_epochs = 200
 
    # --- PRNG setup ---
    root_key = make_prng_key(seed)
    data_key, init_key = split_prng_key(root_key, 2)
 
    # --- Synthetic data ---
    x = sample_input_features(data_key, batch_size, num_features)
    labels = assign_class_labels(x, num_classes)
    y_onehot = one_hot_encode_labels(labels, num_classes)
 
    print("Input shape:", x.shape)
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())
    print("One-hot[0]:", np.asarray(y_onehot[0]).tolist())
 
    # --- Init MLP ---
    layer_sizes = [num_features, hidden_size, hidden_size, num_classes]
    params = init_mlp_params(init_key, layer_sizes, scale=0.1)
    print("Num layers:", len(params))
    print(
        "Layer shapes:",
        [(np.asarray(W).shape, np.asarray(b).shape) for (W, b) in params],
    )
 
    # --- Forward pass before training ---
    logits0 = mlp_forward(params, x)
    loss0 = cross_entropy_loss(logits0, y_onehot)
    acc0 = classification_accuracy(logits0, labels)
    print(f"Initial loss:     {float(loss0):.4f}")
    print(f"Initial accuracy: {float(acc0):.4f}")
 
    # --- Train ---
    trained_params = train_mlp(params, x, y_onehot, learning_rate, num_epochs)
 
    # --- Evaluate ---
    logits_final = mlp_forward(trained_params, x)
    loss_final = cross_entropy_loss(logits_final, y_onehot)
    acc_final = classification_accuracy(logits_final, labels)
    preds = predict_classes(trained_params, x)
 
    print(f"Final loss:       {float(loss_final):.4f}")
    print(f"Final accuracy:   {float(acc_final):.4f}")
    print("Preds[:8]: ", np.asarray(preds[:8]).tolist())
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())