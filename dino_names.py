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
    dinos.shuffle()
        
    TRAIN = True
    USE_LSTM = False

    nn = None

    if USE_LSTM:
        model_name += "_LSTM"
        dinos.prepare_data(False)
        nn = LSTM(hidden_size, dinos, model_name, weights_dir)
    else:
        model_name += "_RNN"
        dinos.prepare_data(True)
        nn = RNN(hidden_size, dinos, model_name, weights_dir)
    EOS_char = '\n'
    if TRAIN:
        nn.optimize(learning_rate, 10000, 0.001)
    else:
        # Generate a few names starting with "L"
        for i in range(100): 
            nn.sample(dinos.encode_seq("L", True)[0], 20, EOS_char)

__main__()