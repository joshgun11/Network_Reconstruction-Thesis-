
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras
tf.compat.v1.disable_eager_execution()


class Kmodel():

    def __init__(self) -> None:
        pass
    
    def main_model(self,num_input):
        loss = keras.losses.BinaryCrossentropy(from_logits=True,name="binary_crossentropy")
        opt = Adam()
        model = Sequential()
        model.add(Dense(64, input_shape = (num_input,), activation = "relu"))
        model.add(Dense(32, activation = "relu"))
        model.add(Dense(1))
        model.compile(optimizer = opt, loss = loss, metrics = ['accuracy'])
        
        return model


if __name__=="__main__":
    model_generator = Kmodel()
    model = model_generator.main_model(24)
    model.summary()
    