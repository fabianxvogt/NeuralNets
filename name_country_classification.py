import os
import numpy as np
from NN.CNN import CNN
from utils.classification_dataset import classification_dataset


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

    # Classification Neural Net erzeugen
    cnn = CNN(hidden_size, ds_names, model_name, weights_dir)


    TRAINING = True

    if (TRAINING):
        # Trainieren
        cnn.optimize(learning_rate)

    else:
        # Validieren
        hits = 0
        hprev = np.zeros((hidden_size, 1))
        for i in range(1000):
            inputs, target = ds_names.get_random_input_and_target(False) # False = Kein Training
            pred = cnn.sample(hprev, inputs)
            name = ""
            for ix in inputs:
                name += ds_names.ix2ch[np.argmax(ix)]
            target_str = ds_names.ix2class[np.argmax(target)]
            print('Name: ' + name)
            print('Actual country: ' + target_str)
            print('Prediction: ' + pred)
            if target_str == pred:
                hits += 1
            print('Accuracy: ' + str(hits/(i+1)) )

main()