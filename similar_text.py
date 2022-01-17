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
    learning_rate = 1e-1
    # Dataset read
    model_name = "shakespeare"

    weights_dir = root + "/weights/"    
    input_file = root + "/dataset/" + model_name
    input_file += ".txt"

    text_data = open(input_file, 'r').read() 
    shakespeare = dataset(text_data, seq_length)

    USE_LSTM = True

    if USE_LSTM:
        shakespeare.encode_data(False)
        lstm = LSTM(hidden_size, shakespeare)
        lstm.optimize(learning_rate, model_name, weights_dir)

    else:
        shakespeare.encode_data(True)
        rnn = RNN(hidden_size, shakespeare)
        rnn.optimize(learning_rate, model_name, weights_dir)

__main__()