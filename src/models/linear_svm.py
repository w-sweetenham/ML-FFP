from curses.ascii import DEL
import numpy as np
import random

DELTA = 0.001

class BinaryLinearSVM:

    def __init__(self, dataset, c):
        self.classes = dataset[:, 0]
        if set(self.classes) != {0, 1}:
            raise ValueError(f'Invalid class indexes: {set(self.classes)}')
        self.num_dims = dataset.shape[-1]-1
        self.c = c
        self.dot_prod_matrix = np.matmul(dataset[:, 1:], dataset[:, 1:].T)
        self.num_datapoints = dataset.shape[0]
        self.w = np.zeros(self.num_dims, dtype=float)
        self.bias = 0
        self.lagrange_multipliers = np.zeros(self.num_datapoints, dtype=float)
        self.x = dataset[:, 1:]
        mapping = lambda x: -1 if x == 0 else 1
        self.t = np.array([mapping(val) for val in dataset[:, 0]])
        self.num_iterations = 0

    def train(self, max_iterations):
        for iteration in range(max_iterations):
            a = random.randint(0, self.num_datapoints-1)
            b = a
            while b == a:
                b = random.randint(0, self.num_datapoints-1)


            second_derriv = self.dot_prod_matrix[a][a] + self.dot_prod_matrix[b][b] - 2*self.dot_prod_matrix[a][b]
            if second_derriv < DELTA:
                self.num_iterations += 1
                continue
            lagrange_a_new = (self.t[a]*(np.dot(self.w, self.x[b]) - self.t[b] - np.dot(self.w, self.x[a]) + self.t[a])/second_derriv) + self.lagrange_multipliers[a]
            gamma = -1.0*(self.t[a]*self.lagrange_multipliers[a] + self.t[b]*self.lagrange_multipliers[b])
            if self.t[a] != self.t[b]:
                L = max(0, gamma*self.t[b])
                H = min(self.c + gamma*self.t[b], self.c)
            else:
                L = max(0, -1*(gamma*self.t[b] + self.c))
                H = min(self.c, -1*gamma*self.t[b])
            
            if lagrange_a_new < L:
                lagrange_a_new = L
            elif lagrange_a_new > H:
                lagrange_a_new = H

            lagrange_b_new = -1*(gamma*self.t[b] + self.t[a]*self.t[b]*lagrange_a_new)

            if lagrange_a_new > DELTA and lagrange_a_new < 1-DELTA:
                bias_a_new = self.bias - (np.dot(self.w, self.x[a]) + self.bias - self.t[a]) \
                    + (self.lagrange_multipliers[a] - lagrange_a_new)*self.t[a]*self.dot_prod_matrix[a][a] \
                    + (self.lagrange_multipliers[b] - lagrange_b_new)*self.t[b]*self.dot_prod_matrix[b][b]
            else:
                bias_a_new = None

            if lagrange_b_new > DELTA and lagrange_b_new < 1-DELTA:
                bias_b_new = self.bias - (np.dot(self.w, self.x[b]) + self.bias - self.t[b]) \
                    + (self.lagrange_multipliers[b] - lagrange_b_new)*self.t[b]*self.dot_prod_matrix[b][b] \
                    + (self.lagrange_multipliers[a] - lagrange_a_new)*self.t[a]*self.dot_prod_matrix[a][a]
            else:
                bias_b_new = None

            if bias_a_new is not None and bias_b_new is not None:
                self.bias = (bias_a_new + bias_b_new)/2
            elif bias_a_new is None and bias_b_new is not None:
                self.bias = bias_b_new
            elif bias_a_new is not None and bias_b_new is None:
                self.bias = bias_a_new


            self.w = self.w + self.t[a]*self.x[a]*(lagrange_a_new - self.lagrange_multipliers[a]) + self.t[b]*self.x[b]*(lagrange_b_new - self.lagrange_multipliers[b])

            self.lagrange_multipliers[a] = lagrange_a_new
            self.lagrange_multipliers[b] = lagrange_b_new

            kkt_broken = False
            for index in range(len(self.lagrange_multipliers)):
                y = self.t[index]*(np.dot(self.w, self.x[index]) + self.bias)
                if abs(self.lagrange_multipliers[index]) < DELTA:
                    if not y > 1-DELTA:
                        kkt_broken = True
                        break
                elif self.lagrange_multipliers[index] > DELTA and self.lagrange_multipliers[index] < self.c-DELTA:
                    if not (y > 1-DELTA and y < 1+DELTA):
                        kkt_broken = True
                        break
                else:
                    if not y < 1+DELTA:
                        kkt_broken = True
                        break
            self.num_iterations += 1
            if not kkt_broken: # kkt conditions held for all datapoints
                print('kkt conditions met')
                break
