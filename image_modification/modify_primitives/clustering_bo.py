# This file implements bayesian optimization and search on the Neural network input space
# Example usage:
# input_dim is the area of search space. If only x, z enter only 2
# BO = bo_class(input_dim=2)
# BO.init_BO(f=classify(generate_image())
# for _ in range(10):
#   BO.run_BO(max_iter=1)
#   BO.bo.suggested_sample
# BO.print_output(fix_output=(0, 0.5)), To fix x at 0.5

import GPyOpt
import numpy as np
import GPy

class bo_class():
    def __init__(self, input_dim):
        self.bounds = [{'name':'x', 'type':'continuous', 'domain':(0,1), 'dimensionality':input_dim}]
        self.input_dim = input_dim

    def init_BO(self, f, kernel=None, constrains=None,  **kwargs):
        self.kernel = kernel
        self.constrains = constrains

        self.bo = GPyOpt.methods.BayesianOptimization(f=f, domain=self.bounds, constrains=self.constrains,
                                                      kernel=self.kernel, **kwargs)

    def run_BO(self, max_iter=10, max_time=np.inf):
        self.bo.run_optimization(max_iter=max_iter, max_time=max_time)
        self.ce()

    def print_output(self, fix_output):
        GPy.plotting.show(self.bo.model.model.plot(fixed_inputs=[fix_output])).show()

    def generate_next_input(self):
        return list(self.bo.suggested_sample)

    def ce(self):
        self.x_opt = self.bo.x_opt
        self.fx_opt = self.bo.fx_opt


