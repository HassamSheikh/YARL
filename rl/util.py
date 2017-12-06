def generator(number):
    return(i for i in range(number))
def clone_model(model):
    import keras
    clone_model = keras.models.clone_model(model)
    clone_model.set_weights(model.get_weights())
    return clone_model
