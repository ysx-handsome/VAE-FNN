import pandas as pd
import numpy as np
import random


class simu_generator():
    def __int__(self,dim,n,sigma_y):
        self.dim=dim
        self.n=n
        self.sigma_y=sigma_y

    def x_to_y(self,x):
        return np.sin(16*x[:,0])

    def generate_data(self):
        x = np.random.randn(self.n, self.dim)
        w = x.copy()
        w[:, 0] = w[:, 0] + np.random.laplace(size=self.n)
        y = self.x_to_y(x)
        yy = y + np.random.normal(size=self.n) * self.sigma_y
        return x, w, y, yy

