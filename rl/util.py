def generator(number):
    return(i for i in range(number))

def clone_model(model):
    import keras
    clone_model=keras.models.clone_model(model)
    clone_model=update_model(clone_model, model)
    return clone_model

def update_model(model_to_be_updated, source):
    model_to_be_updated.set_weights(source.get_weights())
    return model_to_be_updated
