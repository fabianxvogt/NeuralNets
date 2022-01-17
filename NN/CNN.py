import numpy as np

class CNN:
    def __init__(self, hidden_size, dataset, model_name, weights_dir):
        # Dimensions & learning rate
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.output_size = len(dataset.class2ix)

        self.hits = 0

        # Weights and biases
        self.Wxh = np.random.randn(hidden_size, self.dataset.vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # input to hidden
        self.Why = np.random.randn(self.output_size, hidden_size) * 0.01  # input to hidden
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))

        self.model_name = model_name 
        self.weights_dir = weights_dir
        if len(model_name) > 0 and len(weights_dir) > 0:
            self.load_model(model_name, weights_dir)

    # forward pass
    def forward(self, inputs, targets, hs, ys, ps):
        loss = 0    
        for t in range(0, len(inputs)):
          
            hs[t] = np.tanh(
                np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh
            )  # hidden state
            
            ys[t] = (
                np.dot(self.Why, hs[t]) + self.by
            )  # unnormalized log probabilities for next chars
            
            ps[t] = np.exp(ys[t]) / np.sum(
                np.exp(ys[t])
            )  # probabilities for next chars
            tid = np.argmax(targets)
            loss += -np.log(ps[t][tid, 0]) 


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
        tid = np.argmax(targets)
        for t in reversed(range(len(inputs))):
            # output probabilities
            dy = np.copy(ps[t])
            # derive our first gradient
            dy[tid] -= 1  
            # output gradient
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
        hs, ys, ps = {}, {}, {} 
        hs[-1] = np.copy(hprev)

        # Do forward pass and get the loss
        loss, hs, ys, ps = self.forward(inputs, targets, hs, ys, ps)

        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy, dbh, dby = self.backward(inputs, targets, hs, ps)

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def load_model(self, filename, weights_dir):
        # Fixing the path
        if weights_dir[-1] != '/':
            weights_dir += '/'
        print ('(Info) Loading weights for "{}"'.format(filename))
        try:
            self.Wxh = np.load(weights_dir + filename + '_cnn_Wxh' + '.npy')
            self.Whh = np.load(weights_dir + filename + '_cnn_Whh' + '.npy')
            self.bh = np.load( weights_dir + filename + '_cnn_bh' + '.npy')
            self.Why = np.load(weights_dir + filename + '_cnn_Why'  + '.npy')
            self.by = np.load( weights_dir + filename + '_cnn_by'  +'.npy')
        except:
            print('(Error) Can"t find the saved weights!')
            return
        print ('(Info) Weights Loaded successfully!')

    def save_weights(self, filename, weights_dir):
        
        print ('(Info) Saving weights for "{}"'.format(filename))
        np.save(weights_dir + filename + '_cnn_Wxh'    + '.npy', self.Wxh)
        np.save(weights_dir + filename + '_cnn_Whh'    + '.npy', self.Whh)
        np.save(weights_dir + filename + '_cnn_bh'     + '.npy', self.bh)
        np.save(weights_dir + filename + '_cnn_Why'    + '.npy', self.Why)
        np.save(weights_dir + filename + '_cnn_by'     + '.npy', self.by)
        print ('(Info) Weights saved!')

    # Eine Vorhersage samplen
    def sample(self, h, seed):

        x = seed[0]
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        y =  (
                np.dot(self.Why, h) + self.by
            ) 
        # Alle inputs durchgehen
        for t in range(1, len(seed)):

            x = seed[t]
            # hidden state updaten
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            # output (unnormalised)
            y += 1+ (t/len(seed)) * (
                np.dot(self.Why, h) + self.by
            ) 
        # Wahrscheinlichkeiten für Classes berechnen
        p = np.exp(y) / np.sum(
            np.exp(y)
        ) 
        # Auwahl
        ix = np.random.choice(range(self.output_size), p=p.ravel())

        return self.dataset.ix2class[ix]

    def optimize(self, learning_rate):
        

        n = 0
        mWxh, mWhh, mWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(
            self.by
        )  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.output_size) 

        input_counter = 0

        hits = 0

        while True:
            
            hprev = np.zeros((self.hidden_size, 1))
            if  n == 0:
                input_counter = 0 

            inputs, targets = self.dataset.get_random_input_and_target()

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(
                inputs, targets, hprev
            )
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Alle 1000 Iterationen samplen
            if n % 1000 == 0:
                print("iter %d, loss: %f" % (n, smooth_loss)) 
                country_name = ""
                for ix in inputs:
                    country_name += self.dataset.ix2ch[np.argmax(ix)]
                target = self.dataset.ix2class[np.argmax(targets)]
                pred = self.sample(hprev, inputs)
                print('\n\n')
                print('Iter: ' + str(n))
                print('Name: ' + country_name)
                print('Actual Country: ' + target)
                print('Prediction: '+ pred)
                if target == pred:
                    hits += 1
                div = n/1000 +1
                print('Accuracy: ' + str(hits/div) )
                
                self.save_weights(self.model_name, self.weights_dir)

            # Adagrad gradient descent 
            for param, dparam, mem in zip(
                [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                [dWxh, dWhh, dWhy, dbh, dby],
                [mWxh, mWhh, mWhy, mbh, mby],
            ):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            n += 1  # iteration counter
            input_counter += 1