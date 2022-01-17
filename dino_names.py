import os
from NN.LSTM import LSTM
from NN.RNN import RNN

from utils.dataset import dataset

def __main__():

    # get current path
    root = os.path.dirname(os.path.abspath(__file__))

    # Model params
    hidden_size = 100
    seq_length = 25
    learning_rate = 1e-1
    # Dataset read
    model_name = "dinos"

    weights_dir = root + "/weights/"    
    input_file = root + "/dataset/" + model_name
    
    input_file += ".txt"
    text_data = open(input_file, 'r').read() 
    dinos = dataset(text_data, seq_length)

    USE_LSTM = False

    if USE_LSTM:
        dinos.encode_data(False)
        lstm = LSTM(hidden_size, dinos)
        lstm.optimize(learning_rate, model_name, weights_dir)

    else:
        dinos.encode_data(True)
        rnn = RNN(hidden_size, dinos)
        rnn.optimize(learning_rate, model_name, weights_dir)

__main__()