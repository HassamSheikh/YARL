import gym
def generator(number):
    return(i for i in range(number))

def clone_model(model):
    import keras
    clone_model=keras.models.clone_model(model)
    clone_model=clone_weights(clone_model, model)
    return clone_model

def clone_weights(model_to_be_updated, source_model):
    model_to_be_updated.set_weights(source_model.get_weights())
    return model_to_be_updated

def update_model_by_polyak_average(model_to_be_updated, source_model, tau):
    updated_weights=polyak_averaging(model_to_be_updated.get_weights(), source_model.get_weights(), tau)
    model_to_be_updated.set_weights(updated_weights)
    return model_to_be_updated

def polyak_averaging(old, new, tau):
    import numpy as np
    return np.multiply(old, float(tau)) + np.multiply(new, float(1-tau))

def huber_loss(y_true, y_pred, delta_value=1.0):
    import tensorflow as tf
    return tf.losses.huber_loss(y_true, y_pred, delta=delta_value)

def build_gym_environment(gym_env_name):
    """Returns the OpenAI Gym environment with state and action dimensions to build the Keras network"""
    env = gym.make(gym_env_name)
    return env, (*find_state_and_action_dimension(env))

def find_state_and_action_dimension(env):
    """Returns state and action dimensions of the Gym environmentsta"""
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    return state_dim, action_dim
