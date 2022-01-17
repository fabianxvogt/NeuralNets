import numpy as np

class RNN:
    def __init__(self, hidden_size, dataset):
        # Dimensions & learning rate
        self.vocab_size = dataset.vocab_size
        self.hidden_size = hidden_size
        self.dataset = dataset

        # Weights and biases
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # input to hidden
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01  # input to hidden
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

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
            loss += -np.log(ps[t][targets[t][0], 0])  # softmax (cross-entropy loss)

        return loss, hs, ys, ps

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
    def lossFun(self, inputs, targets, hprev):
        hs, ys, ps = {}, {}, {} # Empty dicts
        hs[-1] = np.copy(hprev)

        targets_index = list(np.argmax(targets, axis=1))
        # Do forward pass and get the loss
        loss, hs, ys, ps = self.forward(inputs, targets_index, hs, ys, ps)

        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy, dbh, dby = self.backward(inputs, targets_index, hs, ps)

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def load_model(self, filename, weights_dir):
        # Fixing the path
        if weights_dir[-1] != '/':
            weights_dir += '/'
        print ('(Info) Loading weights for "{}"'.format(filename))
        try:
            self.Wxh = np.load(weights_dir + filename + '_rnn_Wxh' + '.npy')
            self.Whh = np.load(weights_dir + filename + '_rnn_Whh' + '.npy')
            self.bh = np.load( weights_dir + filename +  '_rnn_bh' + '.npy')
            self.Why = np.load(weights_dir + filename +'_rnn_Why'  + '.npy')
            self.by = np.load( weights_dir + filename + '_rnn_by'  +'.npy')
        except:
            print('(Error) Can"t find the saved weights!')
            return
        print ('(Info) Weights Loaded successfully!')

    def save_weights(self, filename, weights_dir):
        
        print ('(Info) Saving weights for "{}"'.format(filename))
        np.save(weights_dir + filename + '_rnn_Wxh'    + '.npy', self.Wxh)
        np.save(weights_dir + filename + '_rnn_Whh'    + '.npy', self.Whh)
        np.save(weights_dir + filename + '_rnn_bh'     + '.npy', self.bh)
        np.save(weights_dir + filename + '_rnn_Why'    + '.npy', self.Why)
        np.save(weights_dir + filename + '_rnn_by'     + '.npy', self.by)
        print ('(Info) Weights saved!')

    # prediction, one full forward pass
    def sample(self, h, seed_ix, n):
        # create vector
        x = np.zeros((self.vocab_size, 1))
        # customize it for our seed char
        x[seed_ix] = 1
        # list to store generated chars
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            # compute output (unnormalised)
            y = np.dot(self.Why, h) + self.by
            ## probabilities for next chars
            p = np.exp(y) / np.sum(np.exp(y))
            # pick one with the highest probability
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)

        txt = "".join(self.dataset.ix2ch[ix] for ix in ixes)
        print("----\n %s \n----" % (txt,))

    def optimize(self, learning_rate, model_name, weights_dir):
        
        self.load_model(model_name, weights_dir)

        n, p = 0, 0
        mWxh, mWhh, mWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(
            self.by
        )  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.dataset.seq_length  # loss at iteration 0

        input_counter = 0

        while n <= 1000 * 100:
            if input_counter >= len(self.dataset.inputs) or n == 0:
                input_counter = 0
                hprev = np.zeros((self.hidden_size, 1))  # reset RNN memory
            inputs = self.dataset.inputs[input_counter]
            targets = self.dataset.targets[input_counter]

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(
                inputs, targets, hprev
            )
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Sample
            if n % 1000 == 0:
                print("iter %d, loss: %f" % (n, smooth_loss))  # print progress
                self.sample(hprev, inputs[0], 200)
                
                self.save_weights(model_name, weights_dir)

            # Adagrad gradient descent
            for param, dparam, mem in zip(
                [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                [dWxh, dWhh, dWhy, dbh, dby],
                [mWxh, mWhh, mWhy, mbh, mby],
            ):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += self.dataset.seq_length  # move data pointer
            n += 1  # iteration counter
            input_counter += 1



