import sampling_primitives as sp

def epsilon_ball(root, epsilon, no_samples, sampling_technique=None):
    f_epsilon = lambda val : root + (val-0.5)*epsilon
    if sampling_technique is None:
        sampling_technique=sp.uniform_sampling
    sampled_points = sampling_technique(len(root), no_samples)
    return [f_epsilon(point) for point in sampled_points]