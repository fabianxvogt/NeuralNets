import numpy as np
from time import time
import aux_funcs as aux
from utils.dataset import dataset
import os

class LSTM:
    def __init__(self, hidden_dim, dataset) -> None:
        self.hidden_dim = hidden_dim
        self.dataset = dataset

        self.Wxh = np.random.randn(dataset.vocab_size, 4*hidden_dim) / np.sqrt(4*hidden_dim)   # input to hidden
        self.Whh = np.random.randn(hidden_dim, 4*hidden_dim) / np.sqrt(4*hidden_dim)  # hidden to hidden
        self.Why = np.random.randn(dataset.vocab_size, hidden_dim) / np.sqrt(hidden_dim)
        self.bh = np.zeros(4*hidden_dim)                                              # hidden bias
        self.by = np.zeros(dataset.vocab_size)

        self.loss = [-np.log(1.0 / dataset.vocab_size)]      # loss 
        
        self.prev_h = np.zeros((1, self.hidden_dim))      # reset LSTM memory

    def sigmoid(self, x):
        return np.exp(x)/(1 + np.exp(x))


    # LSTM functions
    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        _, H = prev_h.shape
        a = prev_h.dot(Wh) + x.dot(Wx) + b      # (1, 4*hidden_dim)
        i = self.sigmoid(a[:, 0:H])
        f = self.sigmoid(a[:, H:2*H])
        o = self.sigmoid(a[:, 2*H:3*H])
        g = np.tanh(a[:, 3*H:4*H])              # (1, hidden_dim)
        next_c = f * prev_c + i * g             # (1, hidden_dim)
        next_h = o * (np.tanh(next_c))          # (1, hidden_dim)
        cache = x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c
        return next_h, next_c, cache


    def lstm_forward(self, x, prev_h, Wxh, Whh, bh):
        cache = []
        prev_c = np.zeros_like(prev_h)
        for i in range(x.shape[0]):     # 0 to seq_length-1
            next_h, next_c, next_cache = self.lstm_step_forward(x[i][None], prev_h, prev_c, Wxh, Whh, bh)
            prev_h = next_h
            prev_c = next_c
            cache.append(next_cache)
            if i > 0:
                h = np.append(h, next_h, axis=0)
            else:
                h = next_h
        return h, cache


    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c = cache
        _, H = dnext_h.shape
        d1 = o * (1 - np.tanh(next_c) ** 2) * dnext_h + dnext_c
        dprev_c = f * d1
        dop = np.tanh(next_c) * dnext_h
        dfp = prev_c * d1
        dip = g * d1
        dgp = i * d1
        do = o * (1 - o) * dop
        df = f * (1 - f) * dfp
        di = i * (1 - i) * dip
        dg = (1 - g ** 2) * dgp
        da = np.concatenate((di, df, do, dg), axis=1)
        db = np.sum(da, axis=0)
        dx = da.dot(Wx.T)
        dprev_h = da.dot(Wh.T)
        dWx = x.T.dot(da)
        dWh = prev_h.T.dot(da)
        return dx, dprev_h, dprev_c, dWx, dWh, db


    def lstm_backward(self, dh, cache):
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        N, H = dh.shape
        dh_prev = 0
        dc_prev = 0
        for i in reversed(range(N)):
            dx_step, dh0_step, dc_step, dWx_step, dWh_step, db_step = self.lstm_step_backward(dh[i][None] + dh_prev, dc_prev, cache[i])
            dh_prev = dh0_step
            dc_prev = dc_step
            if i==N-1:
                dx = dx_step
                dWx = dWx_step
                dWh = dWh_step
                db = db_step
            else:
                dx = np.append(dx_step, dx, axis=0)
                dWx += dWx_step
                dWh += dWh_step
                db += db_step
        dh0 = dh0_step
        return dx, dh0, dWx, dWh, db


    def sample(self, x, h, txt_length, idx_to_char):
        txt = ""
        c = np.zeros_like(h)
        for i in range(txt_length):
            h, c, _ = self.lstm_step_forward(x, h, c, self.Wxh, self.Whh, self.bh)
            scores = np.dot(h, self.Why.T) + self.by
            prob = np.exp(scores) / np.sum(np.exp(scores))
            pred = np.random.choice(range(self.dataset.vocab_size), p=prob[0])
            x = aux.encode([pred], self.dataset.vocab_size)
            next_character = idx_to_char[pred]
            txt += next_character
        return txt

    def load_model(self, filename, weights_dir):
        # Fixing the path
        if weights_dir[-1] != '/':
            weights_dir += '/'
        print ('(Info) Loading weights for "{}"'.format(filename))
        try:
            self.Wxh = np.load(weights_dir + filename + '_Wxh' + '.npy')
            self.Whh = np.load(weights_dir + filename + '_Whh' + '.npy')
            self.bh = np.load( weights_dir + filename +  '_bh' + '.npy')
            self.Why = np.load(weights_dir + filename +'_Why'  + '.npy')
            self.by = np.load( weights_dir + filename + '_by'  +'.npy')
            self.loss = list(np.load(weights_dir + filename + '_loss' +'.npy'))
            self.prev_h = np.load(weights_dir + filename + '_prev_h' + '.npy')
        except:
            print('(Error) Can"t find the saved weights!')
            return
        print ('(Info) Weights Loaded successfully!')

    def save_weights(self, filename, weights_dir):
        print ('(Info) Saving weights for "{}"'.format(filename))

        np.save(weights_dir + filename + '_Wxh'    + '.npy', self.Wxh)
        np.save(weights_dir + filename + '_Whh'    + '.npy', self.Whh)
        np.save(weights_dir + filename + '_bh'     + '.npy', self.bh)
        np.save(weights_dir + filename + '_Why'    + '.npy', self.Why)
        np.save(weights_dir + filename + '_by'     + '.npy', self.by)
        np.save(weights_dir + filename + '_loss'   + '.npy', self.loss)
        np.save(weights_dir + filename + '_prev_h' + '.npy', self.prev_h)
        print ('(Info) Weights saved!')

    def optimize(self, learning_rate, model_name, weights_dir):
        # history variables
        self.loss = [-np.log(1.0 / self.dataset.vocab_size)]      # loss at iteration 0
        
        self.prev_h = np.zeros((1, self.hidden_dim))      # reset LSTM memory

        self.load_model(model_name, weights_dir)

        smooth_loss = self.loss.copy()
        it = 0
        input_counter = 0
        t0 = time()                             # time counting starting here

        while True:
            if input_counter >= len(self.dataset.inputs):
                input_counter = 0
            # collect data for next step
            inputs = self.dataset.inputs[input_counter]
            targets = self.dataset.targets[input_counter]
            
            if it > 0 and it % 100 == 0:
                print('\niter %d, loss: %f' % (it, smooth_loss[-1]))  # print progress
                print("---")
                print(self.sample(inputs[-1], self.prev_h, 1000,self.dataset.ix2ch))
                print("---")
                
                self.save_weights(model_name, weights_dir)
                #aux.plot(loss, smooth_loss, it, it_per_epoch, base_name=data_name)
            it += 1

            # forward pass
            h_states, h_cache = self.lstm_forward(inputs, self.prev_h, self.Wxh, self.Whh, self.bh)                         # (seq_length, hidden_dim)
            scores = np.dot(h_states, self.Why.T) + self.by                                               # (seq_length, input_dim)
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)              # (seq_length, input_dim)
            correct_logprobs = -np.log(probs[range(self.dataset.seq_length), np.argmax(targets, axis=1)])    # (seq_length)
            self.loss.append(np.sum(correct_logprobs) / self.dataset.seq_length)                                  # (1)
            smooth_loss.append(smooth_loss[-1] * 0.999 + self.loss[-1] * 0.001)

            # Backward pass
            dscores = probs
            dscores[range(self.dataset.seq_length), np.argmax(targets, axis=1)] -= 1
            dWhy = dscores.T.dot(h_states)
            dby = np.sum(dscores, axis=0)
            dh_states = dscores.dot(self.Why)
            dx, dh_0, dWx, dWh, db = self.lstm_backward(dh_states, h_cache)

            ### Gradient update ###
            for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWx, dWh, dWhy, db, dby]):
                param -= learning_rate * dparam * 0.5

            prev_h = h_states[-1][None]
            input_counter += 1
        

