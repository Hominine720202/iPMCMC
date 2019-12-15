import numpy as np


def linear_gaussian_state_space(t_max,
                                mu, start_var, transition_var, noise_var,
                                transition_coeffs, observation_coeffs):

    transition = np.random.multivariate_normal(
        np.zeros(mu.shape[0]), transition_var, t_max)
    noise = np.random.multivariate_normal(
        np.zeros(noise_var.shape[0]), noise_var, t_max)

    state = np.random.multivariate_normal(mu, start_var)
    observation = observation_coeffs @ state + noise[0]

    states, observations = [state], [observation]
    for i in range(1, t_max):
        state = transition_coeffs @ state + transition[i]
        observation = observation_coeffs @ state + noise[i]
        states.append(state)
        observations.append(observation)
    return np.array(states), np.array(observations)


def nonlinear_gaussian_state_space(t_max, mu,
                                   noise_std, transition_std, start_std):
    transition = np.random.normal(
        0, transition_std, t_max)
    noise = np.random.normal(
        0, noise_std, t_max)

    state = np.random.normal(mu, start_std)
    observation = (state**2)/20 + noise[0]

    states, observations = [state], [observation]
    for i in range(1, t_max):
        state = state/2 + 25*(state/(1+state**2)) + 8 * \
            np.cos(1.2*(i+1))*transition[i]
        observation = (state**2)/20 + noise[i]
        states.append(state)
        observations.append(observation)
    return np.array(states), np.array(observations)
