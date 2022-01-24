import numpy as np
from time import time

from NN.LSTM import LSTM
from utils.plots import plot_accuracy_and_loss

class classification_LSTM(LSTM):
    def __init__(self, hidden_size, dataset, model_name, weights_dir):
        super().__init__(hidden_size, dataset, model_name, weights_dir)
        self.hits = 0
        self.accuracy_over_time = []
        self.loss_over_time = []
        self.epochs = []

    def sample(self, seed):
        self.hidden_state = self.init_hidden_state()
        self.cell_state = np.zeros_like(self.hidden_state)
        
        # Alle inputs durchgehen
        for t in range(0, len(seed)):
            pred = self.get_prediction(seed[t])
        return self.dataset.ix2class[pred]

    def print_sample(self, inputs, target, it, sample_steps):
        pred = self.sample(inputs)
        print('\n\n')
        print('\niter %d, loss: %f' % (it, self.smooth_loss[-1]))  # print progress

        input_txt = ""
        for ix in inputs:
            input_txt += self.dataset.ix2ch[np.argmax(ix)]
        target = self.dataset.ix2class[np.argmax(target)]
        
        print('Name: ' + input_txt)
        print('Actual Country: ' + target)
        print('Prediction: '+ pred)
        if target == pred:
            self.hits += 1
        div = it/sample_steps +1
        acc = self.hits/div
        print('Accuracy: ' + str(acc) )
        self.accuracy_over_time.append(acc)
        self.loss_over_time.append(self.smooth_loss[-1])
        self.epochs.append(it)

    def after_iteration(self, iteration):
        if iteration % 200000 != 0 or iteration == 0:
            return
        plot_accuracy_and_loss(
            self.epochs,
            self.accuracy_over_time,
            self.loss_over_time
        )