from xml.dom.minidom import Node
from models import Kmodel
from parsers import KParseArgs
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from preapre_data import KData
import sys
class KTrain():

    def __init__(self) -> None:
        pass



    def train(self,args,node):

        model_generator = Kmodel()
        data_selector = KData()
       
        
        
        x,y = data_selector.single_experiment(args.data,node)
        model = model_generator.main_model(x.shape[1])

        callbacks = [EarlyStopping(monitor="val_loss",patience=15,verbose=1,
                           mode="auto",restore_best_weights=True)]
        
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .33,random_state = 7,stratify = y)
        
        history = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = args.epochs,
                    batch_size = args.batch_size,callbacks = callbacks)

        return x,y,model

    

if __name__=="__main__":
    train_model = KTrain()
    parser = KParseArgs()
    args = parser.parse_args()

    flag = len(sys.argv) == 1

    train_model.train(args,args.node)

        