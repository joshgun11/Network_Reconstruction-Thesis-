
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras
tf.compat.v1.disable_eager_execution()


class Kmodel():

    def __init__(self) -> None:
        pass
    
    def main_model(self,num_input):
        loss = tf.keras.losses.CategoricalCrossentropy()
        opt = Adam(learning_rate = 0.001)
        model = Sequential()
        model.add(Dense(100, input_shape = (num_input,), activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, input_shape = (num_input,), activation = "relu"))
        model.add(Dense(32, activation = "relu"))
        model.add(Dense(2,activation = 'softmax'))
        model.compile(optimizer = opt, loss = loss, metrics = ['accuracy'])
        
        return model


if __name__=="__main__":
    model_generator = Kmodel()
    model = model_generator.main_model(24)
    model.summary()
    