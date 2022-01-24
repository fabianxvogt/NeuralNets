from abc import abstractmethod
import py_compile


import numpy as np

class RNN:
    def __init__(self, hidden_size, dataset, model_name, weights_dir):
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.model_name = model_name
        self.weights_dir = weights_dir

        self.INPUTS_AS_VECTORS = True
        # Weights and biases
        (self.Wxh, self.Whh, self.Why, self.bh, self.by) = self.init_weights()

        # Hidden state
        self.hidden_state = self.init_hidden_state()
        
        # Loss
        self.smooth_loss = -np.log(1.0 / self.dataset.input_size) * self.dataset.seq_length  # loss at iteration 0

        # Used for Adagrad
        self.mWxh, self.mWhh, self.mWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(
            self.by
        )  #
        self.load_model(model_name, weights_dir)

    # Initialize the weights
    def init_weights(self):
        
        return (np.random.randn(self.hidden_size, self.dataset.input_size) * 0.01,  # input to hidden
                np.random.randn(self.hidden_size, self.hidden_size) * 0.01,  # input to hidden
                np.random.randn(self.dataset.output_size, self.hidden_size) * 0.01,  # input to hidden
                np.zeros((self.hidden_size, 1)),
                np.zeros((self.dataset.output_size, 1)),
        )
    
    def init_hidden_state(self):
        return np.zeros((self.hidden_size, 1))

    def load_model(self, filename, weights_dir):
        # Fixing the path
        if weights_dir[-1] != '/':
            weights_dir += '/'
        print ('(Info) Loading weights for "{}"'.format(filename))
        try:
            self.Wxh = np.load(weights_dir + filename + '_'+'_Wxh'+ '.npy')
            self.Whh = np.load(weights_dir + filename + '_'+'_Whh'+ '.npy')
            self.bh  = np.load(weights_dir + filename + '_'+'_bh' + '.npy')
            self.Why = np.load(weights_dir + filename + '_'+'_Why'+ '.npy')
            self.by  = np.load(weights_dir + filename + '_'+'_by' +'.npy')
        except:
            print('(Error) Can"t find the saved weights!')
            return
        print ('(Info) Weights Loaded successfully!')

    def save_model(self, filename, weights_dir):
        
        print ('(Info) Saving weights for "{}"'.format(filename))
        np.save(weights_dir + filename + '_'+'_Wxh'    + '.npy', self.Wxh)
        np.save(weights_dir + filename + '_'+'_Whh'    + '.npy', self.Whh)
        np.save(weights_dir + filename + '_'+'_bh'     + '.npy', self.bh)
        np.save(weights_dir + filename + '_'+'_Why'    + '.npy', self.Why)
        np.save(weights_dir + filename + '_'+'_by'     + '.npy', self.by)
        print ('(Info) Weights saved!')

    def gradient_descent(self, dWxh, dWhh, dWhy, dbh, dby, learning_rate):
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) 

        # Adagrad gradient descent 
        for param, dparam, mem in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
        ):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
        
    def compute_loss(self, targets, ps):
        loss = 0
        for i in range(len(ps)):
            loss += -np.log(ps[i][targets[i][0], 0])
        return loss

    # forward pass
    def forward(self, inputs, targets, hs, ys, ps):
        loss = 0

        for t in range(len(inputs)):
            hs[t] = np.tanh(
                np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh
            )  # hidden state
            ys[t] = (
                np.dot(self.Why, hs[t]) + self.by
            )  # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(
                np.exp(ys[t])
            )  # probabilities for next chars
        return hs, ys, ps

    # backward pass
    def backward(self, inputs, targets, hs, ps):
        # initalize vectors for gradient values for each set of weights
        dWxh, dWhh, dWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            # output probabilities
            dy = np.copy(ps[t])
            # derive our first gradient
            if len(targets) == 1: # if this RNN is used for classifications, targets will have length = 1 (same target for all inputs)
                dy[targets[0]] -= 1  # backprop into y
            else:
                dy[targets[t]] -= 1  # backprop into y
            dWhy += np.dot(dy, hs[t].T)
            # derivative of output bias
            dby += dy
            # backpropagate!
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw  # derivative of hidden bias
            dWxh += np.dot(dhraw, inputs[t].T)  # derivative of input to hidden layer weight
            dWhh += np.dot(
                dhraw, hs[t - 1].T
            )  # derivative of hidden layer to hidden layer weight
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        return dWxh, dWhh, dWhy, dbh, dby

    # loss function 
    def loss_fun(self, inputs, targets, hprev):
        hs, ys, ps = {}, {}, {} # Empty dicts
        hs[-1] = np.copy(hprev)

        targets_index = list(np.argmax(targets, axis=1))
        # Do forward pass and get the loss
        hs, ys, ps = self.forward(inputs, targets_index, hs, ys, ps)

        loss = self.compute_loss(targets_index, ps)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy, dbh, dby = self.backward(inputs, targets_index, hs, ps)

        self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

        return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]
    
    def get_prediction(self, input):
        self.hidden_state = np.tanh(np.dot(self.Wxh, input) + np.dot(self.Whh, self.hidden_state) + self.bh)
        # compute output (unnormalised)
        y = np.dot(self.Why, self.hidden_state) + self.by
        ## probabilities for next chars
        p = np.exp(y) / np.sum(np.exp(y))
        # pick one with the highest probability
        return np.random.choice(range(self.dataset.output_size), p=p.ravel())
            

    def sample(self, input, n = 10000, EOS_char = None):
        txt = self.dataset.ix2ch[np.argmax(input)]
        
        for t in range(n):
            ix = self.get_prediction(input)
            if not EOS_char is None:
                if (self.dataset.ix2ch[ix] == EOS_char):
                    break
            ch = self.dataset.ix2ch[ix]
            txt += ch
            # Use as new input
            input = self.dataset.encode_seq(ch, self.INPUTS_AS_VECTORS)[0]
        return txt

    def print_sample(self, inputs, targets, iteration, sample_steps):
        new_text = self.sample(inputs[0])
        print("iter %d, loss: %f" % (iteration, self.smooth_loss))  # print progress     
        print(new_text)

    def after_iteration(self, iteration):
        return

    def optimize(self, learning_rate, n, sample_rate):       

        sample_steps = 1/sample_rate
        for i in range(n):
            inputs, targets = self.dataset.get_next_inputs_and_targets(self.INPUTS_AS_VECTORS)
            
            dWxh, dWhh, dWhy, dbh, dby, self.hidden_state = self.loss_fun(
                inputs, targets, self.hidden_state
            )

            # Sample
            if i % sample_steps == 0:
                self.print_sample(inputs, targets, i, sample_steps)
                
                self.save_model(self.model_name, self.weights_dir)

            self.gradient_descent(dWxh, dWhh, dWhy, dbh, dby, learning_rate)
            self.after_iteration(i)