import numpy as np
from utils.dataset import dataset


class classification_dataset(dataset):
    def __init__(self, classified_text_data) -> None:
        full_text = ""
        self.class2ix = {}
        self.ix2class = {}
        self.class_data = {}

        self.train_data = {}
        self.valid_data = {}

        # define class mapping and concat all texts for the base class
        i = 0
        for class_key in classified_text_data:
            # Set Output Mappings
            self.class2ix[class_key] = i
            self.ix2class[i] = class_key

            class_text = classified_text_data[class_key]
            full_text+= class_text
            self.class_data[class_key] = class_text.splitlines(True)
            i += 1
    
        super().__init__(full_text, 0)
        
        self.output_size = len(self.class2ix)

    def prepare_data(self, as_vectors, training_ratio = 0.8):

        train_data = {}
        valid_data = {}
        counter = 0
        for cls in self.class_data:
            values = self.class_data[cls]
            
            #np.random.shuffle(names)     # Das war ein Versuch, aber die Namen sollten hier noch nicht 
            # geshuffelt werden, sonst sind die Trainings- und Validierungsdaten bei jeden Start verschieden

            encoded_names_train = []
            encoded_names_valid = []
            name_counter = 0
            for v in values:
                encoded_name = self.encode_seq(v,as_vectors)
                name_counter += 1
                if name_counter > training_ratio*len(values): 
                    encoded_names_valid.append(encoded_name)
                else:
                    encoded_names_train.append(encoded_name)
            train_data[counter] = encoded_names_train
            valid_data[counter] = encoded_names_valid
            counter += 1
        return train_data, valid_data

    def get_next_inputs_and_targets(self, as_vectors = False, training=True):
        if(len(self.train_data) == 0):
            self.train_data, self.valid_data = self.prepare_data(as_vectors)

        if training:
            dataset = self.train_data
        else:
            dataset = self.valid_data

        # generate random country
        class_ix = np.random.randint(low=0, high=len(dataset))
        # generate random name
        element_ix = np.random.randint(low=0, high=len(dataset[class_ix]))

        enc = np.zeros(len(dataset), dtype=int)
        enc[class_ix] = 1
        return dataset[class_ix][element_ix], [enc]


