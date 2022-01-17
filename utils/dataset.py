import numpy as np


class dataset:
    def __init__(self, text_data, seq_length) -> None:
        self.seq_length = seq_length
        self.data = text_data
        self.chars = sorted(set(self.data))
        self.vocab_size = len(self.chars)
        self.ch2ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix2ch = {i: ch for i, ch in enumerate(self.chars)}

        # will be set later on
        self.encoded_data = None
        self.inputs = []
        self.targets = []

    def get_inputs_and_targets(self, as_vectors = False):
        if self.encoded_data == None:
            self.encode_data(as_vectors)
        return self.inputs, self.targets

    def encode_data(self,as_vectors):
        encoded_data = []
        inputs = []
        targets = []
        seq_counter = 0
        while(True):
            if seq_counter + self.seq_length >= len(self.data):
                break
            #self.seq_counter = 0  # go to start of data
            x = [self.ch2ix[char] for char in self.data[seq_counter: seq_counter + self.seq_length+1]]
            seq_counter += self.seq_length
            encoded_seq = self.__encode(x,as_vectors)
            encoded_data.append(encoded_seq)
            input = encoded_seq[:len(encoded_seq)-1]
            target = encoded_seq[1:]
            inputs.append(input)
            targets.append(target)
        self.encoded_data = encoded_data
        self.inputs = inputs
        self.targets = targets

    def __encode(self, seq, as_vector):   
        # 1-of-k encoding
        if as_vector:
            enc = np.zeros((1, self.vocab_size, 1), dtype=int)
        else:
            enc = np.zeros((1, self.vocab_size), dtype=int)
        
        enc[0][seq[0]] = 1
        for i in range(1, len(seq)):
            if as_vector:
                row = np.zeros((1, self.vocab_size, 1), dtype=int)
            else:
                row = np.zeros((1, self.vocab_size), dtype=int)
            row[0][seq[i]] = 1
            enc = np.append(enc, row, axis=0)
        return enc

    # a) 1. Input-Tokens in Vektoren mit fester Länge umwandeln
    # ch2ix: Mapping von Buchstaben auf Integerwerte
    # seq: Text-sequenz, die umgewandelt werden soll
    def encode_seq(self, seq):
        name_ix = [self.ch2ix[char] for char in seq]
        encoded_name = []
        for char_ix in name_ix:
            enc = np.zeros((self.vocab_size, 1), dtype=int)
            enc[char_ix][0] = 1
            encoded_name.append(enc)
        return encoded_name



    
