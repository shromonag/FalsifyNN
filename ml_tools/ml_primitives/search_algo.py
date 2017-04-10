import sampling_primitives as sp

def basis_vecs(dim, epsilon):
    return [basis_vec(i, dim, epsilon) for i in range(dim)]

def basis_vec(i, dim, epsilon):
    a = np.zeros(dim)
    a[i] = epsilon
    return a

def epsilon_ball(root, epsilon, no_samples, sampling_technique=None):
    f_epsilon = lambda val : root + (val-0.5)*epsilon
    if sampling_technique is None:
        sampling_technique=sp.uniform_sampling
    sampled_points = sampling_technique(len(root), no_samples)
    return [f_epsilon(point) for point in sampled_points]

def gradient_descent(root, epsilon, cost, n_iter):
    bv = basis_vecs(len(root), epsilon)
    for _ in range(n_iter):
        ith = np.array([cost(root+i_val) for i_val in bv]).argmin()
        root = root + bv[ith]
    return root




