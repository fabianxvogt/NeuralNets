import os
import numpy as np
from NN.classification_RNN import classification_RNN
from NN.classification_LSTM import classification_LSTM
from utils.classification_dataset import classification_dataset
import matplotlib.pyplot as plt


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # Model params
    hidden_size = 80
    learning_rate = 1e-1
    # Dataset read
    model_name = "names"

    weights_dir = root + "/weights/"    
    input_file = root + "/dataset/" + model_name

    # Texte aller Dateien in ein Dict packen
    names_data_txt = {}
    filenames = next(os.walk(input_file), (None, None, []))[2]  # [] if no file

    for filename in filenames:
        fname = filename.replace(".txt", "")
        full_path = input_file + "/" + filename
        text_data = open(full_path, mode="r", encoding="utf-8").read()
        names_data_txt[fname] = text_data

    # Dataset erstellen
    ds_names = classification_dataset(names_data_txt)

    USE_LSTM = False
    TRAINING = True
    # Classification Neural Net erzeugen
    if USE_LSTM == True:
        model_name += "_LSTM"
        nn = classification_LSTM(hidden_size, ds_names, model_name, weights_dir)
    else:
        model_name += "_RNN"
        nn = classification_RNN(hidden_size, ds_names, model_name, weights_dir)


    if (TRAINING):
        # Trainieren
        nn.optimize(learning_rate,1000000,0.001)
        
    else:
        print(nn.sample(ds_names.encode_seq("Fabian",not USE_LSTM)))
        print(nn.sample(ds_names.encode_seq("Louise",not USE_LSTM)))

        # Validieren
        hits = 0
        acc_over_time = []
        for i in range(1000):
            inputs, target = ds_names.get_next_inputs_and_targets(not USE_LSTM, False) # False = Kein Training
            pred = nn.sample(inputs)
            name = ""
            for ix in inputs:
                name += ds_names.ix2ch[np.argmax(ix)]
            target_str = ds_names.ix2class[np.argmax(target)]
            print('Name: ' + name)
            print('Actual country: ' + target_str)
            print('Prediction: ' + pred)
            
            if target_str == pred:
                hits += 1
            acc = hits/(i+1)
            acc_over_time.append(acc)
            print('Accuracy: ' + str(acc) )
        
        plt.plot(acc_over_time)
        plt.show()

main()