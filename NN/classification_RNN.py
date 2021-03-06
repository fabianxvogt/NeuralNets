import collections
import numpy as np
from NN.RNN import RNN
from pandas import DataFrame
from utils.plots import plot_accuracy_and_loss

class classification_RNN(RNN):
    def __init__(self, hidden_size, dataset, model_name, weights_dir):
        super().__init__(hidden_size, dataset, model_name, weights_dir)
        self.hits = 0
        self.accuracy_over_time = []
        self.loss_over_time = []
        self.epochs = []
        
        # Confusion matrix init
        self.confusion = collections.defaultdict(dict)
        for target in self.dataset.class_data:
            for pred in self.dataset.class_data:
                self.confusion[target][pred] = 0

    def set_and_print_confusion(self, target, pred):
        self.confusion[target][pred] +=1
        df = DataFrame(self.confusion).T
        df.fillna(0, inplace=True)
        print("Y (vertical) = Targets; X (horizontal) = Predictions")
        print(df)

    def compute_loss(self, target, ps):
        loss = 0
        for i in range(len(ps)):
            loss += -np.log(ps[i][target[0], 0])
        return loss
    
    def sample(self, inputs):
        self.hidden_state = self.init_hidden_state()
        # Alle inputs durchgehen
        for t in range(0, len(inputs)):
            pred = self.get_prediction(inputs[t])
        return self.dataset.ix2class[pred]

    def print_sample(self, inputs, targets, iteration, sample_steps):
        print("iter %d, loss: %f" % (iteration, self.smooth_loss)) 
        country_name = ""
        for ix in inputs:
            country_name += self.dataset.ix2ch[np.argmax(ix)]
        target = self.dataset.ix2class[np.argmax(targets)]
        pred = self.sample(inputs)
        print('\n\n')
        print('Iter: ' + str(iteration))
        print('Name: ' + country_name)
        print('Actual Country: ' + target)
        print('Prediction: '+ pred)
        if target == pred:
            self.hits += 1
        div = iteration/sample_steps +1
        acc = self.hits/div
        print('Accuracy: ' + str(acc) )
        self.accuracy_over_time.append(acc)
        self.loss_over_time.append(self.smooth_loss)
        self.epochs.append(iteration)

        self.set_and_print_confusion(target, pred)

    
    def after_iteration(self, iteration):
        if iteration % 10000000 != 0 or iteration == 0:
            return
        plot_accuracy_and_loss(
            self.epochs,
            self.accuracy_over_time,
            self.loss_over_time
        )
        # Alle 100k Iterationen Wahrscheinlichkeit plotten
        