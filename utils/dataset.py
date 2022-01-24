from ntpath import join
import numpy as np


class dataset:
    def __init__(self, text_data, seq_length) -> None:
        self.seq_length = seq_length
        self.data = text_data
        self.chars = sorted(set(self.data))
        self.vocab_size = len(self.chars)
        self.ch2ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix2ch = {i: ch for i, ch in enumerate(self.chars)}
        self.input_size = self.vocab_size
        self.output_size = self.vocab_size

        # will be set later on
        self.encoded_data = None
        self.inputs = []
        self.targets = []

        self.n = 0
    
    def get_next_inputs_and_targets(self, as_vectors):
        if self.n>len(self.inputs)-1:
            self.n = 0
        i, t = self.inputs[self.n], self.targets[self.n]
        self.n += 1
        return i, t

    def shuffle(self):
        lines = self.data.splitlines(True)
        np.random.shuffle(lines)
        self.data = ''.join(lines)

    def prepare_data(self,as_vectors):
        encoded_data = []
        inputs = []
        targets = []
        seq_counter = 0
        while(True):
            if seq_counter + self.seq_length >= len(self.data):
                break
            #self.seq_counter = 0  # go to start of data
            #x = [self.ch2ix[char] for char in self.data[seq_counter: seq_counter + self.seq_length+1]]
            x = [char for char in self.data[seq_counter: seq_counter + self.seq_length+1]]
            seq_counter += self.seq_length
            encoded_seq = self.encode_seq(x, as_vectors)#self.encode(x, as_vectors)# encode_seq(x)#,as_vectors)
            encoded_data.append(encoded_seq)
            input = encoded_seq[:len(encoded_seq)-1]
            target = encoded_seq[1:]
            inputs.append(input)
            targets.append(target)
        self.encoded_data = encoded_data
        self.inputs = inputs
        self.targets = targets

    # a) 1. Input-Tokens in Vektoren mit fester Länge umwandeln
    # ch2ix: Mapping von Buchstaben auf Integerwerte
    # seq: Text-sequenz, die umgewandelt werden soll
    def encode_seq(self, seq, as_vector = False):
        if(as_vector):
            dim = (1, self.vocab_size, 1)
        else:
            dim = (1, self.vocab_size)

        encoded_name = np.zeros(dim, dtype=int)
        encoded_name[0][self.ch2ix[seq[0]]] = 1
        for i in range(1,len(seq)):
            enc = np.zeros(dim, dtype=int)
            enc[0][self.ch2ix[seq[i]]] = 1
            encoded_name = np.append(encoded_name,enc, axis=0)            
        return encoded_name





    
