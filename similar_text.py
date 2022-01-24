import os
from NN.LSTM import LSTM
from NN.RNN import RNN

from utils.dataset import dataset

def __main__():

    # get current path
    root = os.path.dirname(os.path.abspath(__file__))

    # Model params
    hidden_size = 250
    seq_length = 25
    learning_rate = 1e-2
    # Dataset read
    model_name = "shakespeare"

    weights_dir = root + "/weights/"    
    input_file = root + "/dataset/" + model_name
    input_file += ".txt"

    text_data = open(input_file, 'r').read() 
    shakespeare = dataset(text_data, seq_length)


    USE_LSTM = False

    if USE_LSTM:
        model_name += "_LSTM"
        shakespeare.prepare_data(False)
        nn = LSTM(hidden_size, shakespeare, model_name, weights_dir)
    else:
        model_name += "_RNN"
        shakespeare.prepare_data(True)
        nn = RNN(hidden_size, shakespeare, model_name, weights_dir)
    
    nn.optimize(learning_rate, 100000000, 0.01)

__main__()