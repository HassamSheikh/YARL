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
