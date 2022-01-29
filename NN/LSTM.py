import numpy as np
from time import time

from NN.RNN import RNN

class LSTM(RNN):
    def __init__(self, hidden_size, dataset, model_name, weights_dir):
        super().__init__(hidden_size, dataset, model_name, weights_dir)        
        self.loss = [-np.log(1.0 / dataset.input_size)]      # loss 
        self.smooth_loss = self.loss.copy()

        self.cell_state = np.zeros_like(self.hidden_state)
        self.INPUTS_AS_VECTORS = False


    def init_weights(self):
        r = (
            np.random.randn(self.dataset.input_size, 4*self.hidden_size) / np.sqrt(4*self.hidden_size),  
            np.random.randn(self.hidden_size, 4*self.hidden_size) / np.sqrt(4*self.hidden_size),            
            np.random.randn(self.dataset.output_size, self.hidden_size) / np.sqrt(self.hidden_size),
            np.zeros(4*self.hidden_size),                                            
            np.zeros(self.dataset.output_size)
        )
        return r

    def init_hidden_state(self):
        return np.zeros((1, self.hidden_size))  


    def sigmoid(self, x):
        return np.exp(x)/(1 + np.exp(x))

    # LSTM functions
    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        _, H = prev_h.shape
        a = prev_h.dot(Wh) + x.dot(Wx) + b     
        i = self.sigmoid(a[:, 0:H])
        f = self.sigmoid(a[:, H:2*H])
        o = self.sigmoid(a[:, 2*H:3*H])
        g = np.tanh(a[:, 3*H:4*H])           
        next_c = f * prev_c + i * g             
        next_h = o * (np.tanh(next_c))          
        cache = x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c
        return next_h, next_c, cache


    def lstm_forward(self, x, prev_h, Wxh, Whh, bh):
        cache = []
        prev_c = np.zeros_like(prev_h)

        for i in range(len(x)):#.shape[0]):     # 0 to seq_length-1
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
    
    def loss_fun(self, inputs, targets, hprev):
        # forward pass
        h_states, h_cache = self.lstm_forward(inputs, hprev, self.Wxh, self.Whh, self.bh)                        
        scores = np.dot(h_states, self.Why.T) + self.by                                              
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)              
        correct_logprobs = -np.log(probs[range(len(inputs)), np.argmax(targets)])    
        self.loss.append(np.sum(correct_logprobs) / len(inputs))                                  
        self.smooth_loss.append(self.smooth_loss[-1] * 0.999 + self.loss[-1] * 0.001)

        # Backward pass
        dscores = probs
        dscores[range(len(inputs)), np.argmax(targets, axis=1)] -= 1
        dWhy = dscores.T.dot(h_states)
        dby = np.sum(dscores, axis=0)
        dh_states = dscores.dot(self.Why)
        dx, dh_0, dWxh, dWhh, dbh = self.lstm_backward(dh_states, h_cache)

        return dWxh, dWhh, dWhy, dbh, dby, h_states[-1][None]#len(inputs) - 1][None]

    def get_prediction(self, input):
        self.hidden_state, self.cell_state, _ = self.lstm_step_forward(input, self.hidden_state, self.cell_state, self.Wxh, self.Whh, self.bh)
        scores = np.dot(self.hidden_state, self.Why.T) + self.by
        prob = np.exp(scores) / np.sum(np.exp(scores))
        return np.random.choice(range(self.dataset.output_size), p=prob[0])

    def print_sample(self, inputs, targets, it, sample_steps): 
        sample = self.sample(inputs[0][None])#,self.hidden_state)
        print('\niter %d, loss: %f' % (it, self.smooth_loss[-1]))  # print progress
        print("---")
        print(sample)
        print("---")