import numpy as np
import ghalton
import funcy as fn

prime_no = [2, 3, 5, 7, 11, 13, 17, 19, 23, 39, 31, 37, 41, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 91, 97]
uniform_sampling = lambda no_random, no_samples: np.random.random_sample((no_samples, no_random))

get_perm_list = lambda no_random: [np.random.choice(prime_no[i], size = prime_no[i], replace=False) for i in range(no_random)]
i_lattice = lambda i, k, irr_nos: fn.flatten([np.true_divide(i, k), lattice_points(i, irr_nos)])
lattice_points = lambda i, irr_nos: [(i*j)%1 for j in irr_nos]


def halton_sampling(no_random, no_samples):
    perm = get_perm_list(no_random)
    seq = ghalton.GeneralizedHalton(perm)
    return seq.get(no_samples)

def lattice_sampling(no_random, no_samples, k):
    if k is None:
        k = no_samples
    irrational_nums = [(np.sqrt(prime_no[i]) + 1)/2 for i in range(1, no_random)]
    return [i_lattice(i+1, k, irrational_nums) for i in range(no_samples)]