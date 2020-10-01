from ptype.utils import (
    normalize_log_probs,
    log_weighted_sum_probs,
    log_weighted_sum_normalize_probs,
)
from copy import copy
from scipy import optimize
import numpy as np

Inf = np.Inf


def vecnorm(x, ord=2):
    if ord == Inf:
        return np.amax(np.abs(x))
    elif ord == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x) ** ord, axis=0) ** (1.0 / ord)


LOG_EPS = -1e150


class Model:
    TYPE_INDEX = 0
    MISSING_INDEX = 1
    ANOMALIES_INDEX = 2
    LLHOOD_TYPE_START_INDEX = 2

    def __init__(
        self, types, df=None, training_params=None, PI=[0.98, 0.01, 0.01],
    ):
        self.types = types
        self.df = df
        self.PI = PI  # weight of pi variable
        if training_params is not None:
            self.training_params = training_params
            self.current_runner = copy(training_params.current_runner)
            self.unique_vals = np.concatenate([np.unique(df.values) for df in training_params.dfs])
            self.dfs_unique_vals_counts = self.get_unique_vals_counts(training_params.dfs)
            self.current_runner.set_unique_values(self.unique_vals)
            self.K = len(self.current_runner.machines) - 2
            self.pi = [self.PI for _ in range(self.K)]

    def get_unique_vals_counts(self, dfs):
        # Finding unique values and their counts
        return {i: {col: [vs, np.array(counts)]
                    for col, (vs, counts) in {col: np.unique(df[col].tolist(), return_counts=True)
                                              for col in df.columns}.items()}
                for i, df in enumerate(dfs)}

    ###################### MAIN METHODS #######################
    def run_inference(self, logP, counts):
        # Constants
        I, J = logP.shape  # I: num of rows in a data column.
        # J: num of data types including missing and catch-all
        K = (
            J - 2
        )  # K: num of possible column data types (excluding missing and catch-all)

        # Initializations
        pi = [self.PI for j in range(K)]  # mixture weights of row types

        # Inference
        p_t = []  # p_t: posterior probability distribution of column types
        p_z = np.zeros(
            (I, K, 3)
        )  # p_z: posterior probability distribution of row types

        counts_array = np.array(counts)

        # Iterates for each possible column type
        for j in range(K):

            # Sum of weighted likelihoods (log-domain)
            p_t.append(
                (
                    counts_array
                    * log_weighted_sum_probs(
                        pi[j][0],
                        logP[:, j + self.LLHOOD_TYPE_START_INDEX],
                        pi[j][1],
                        logP[:, self.MISSING_INDEX - 1],
                        pi[j][2],
                        logP[:, self.ANOMALIES_INDEX - 1],
                    )
                ).sum()
            )

            # Calculates posterior cell probabilities

            # Normalizes
            x1, x2, x3, log_mx, sm = log_weighted_sum_normalize_probs(
                pi[j][0],
                logP[:, j + self.LLHOOD_TYPE_START_INDEX],
                pi[j][1],
                logP[:, self.MISSING_INDEX - 1],
                pi[j][2],
                logP[:, self.ANOMALIES_INDEX - 1],
            )

            p_z[:, j, self.TYPE_INDEX] = np.exp(x1 - log_mx - np.log(sm))
            p_z[:, j, self.MISSING_INDEX] = np.exp(x2 - log_mx - np.log(sm))
            p_z[:, j, self.ANOMALIES_INDEX] = np.exp(x3 - log_mx - np.log(sm))
            p_z[:, j, :] = p_z[:, j, :] / p_z[:, j, :].sum(axis=1)[:, np.newaxis]

        p_t = normalize_log_probs(np.array(p_t))
        self.p_t = {t: p for t, p in zip(self.types, p_t)}
        self.p_z = p_z

    def update_PFSMs(self, runner):
        w_j_z = self.get_all_parameters_z(runner)

        # Find new values using Conjugate Gradient method
        w_j_z, j = self.conjugate_gradient(w_j_z)

        new_runner = copy(runner)
        new_runner, _ = new_runner.set_all_probabilities_z(w_j_z, normalize=True)

        return new_runner

    def conjugate_gradient(self, w, J=10, gtol=1e-5):
        d, g = [], []

        gnorm = gtol + 1
        j = 0
        while gnorm > gtol and j < J:
            if j == 0:
                g.append(self.g_cols(w))
                d.append(-g[j])

            res = optimize.line_search(
                self.f_cols, self.g_cols, w, d[j], g[j], self.f_cols(w)
            )
            if res[0] is None:
                return w, j
            else:
                alpha = res[0]
                w = w + alpha * d[j]

                g.append(self.g_cols(w))
                gnorm = vecnorm(g[j + 1], ord=np.Inf)

                beta_j = max(
                    0, np.dot(g[j + 1].T, g[j + 1] - g[j]) / np.dot(g[j], g[j])
                )  # eq. 7.74 Polak-Ribiere
                d.append(-g[j + 1] + beta_j * d[j])  # eq.7.67

                j += 1

        return w, j

    def f_col(self, i_, column_name, y_i):
        [temp_x, counts_array] = self.dfs_unique_vals_counts[i_][column_name]
        logP = np.array([self.all_probs[str(x_i)] for x_i in temp_x])
        q = []
        for k in range(self.K):
            q.append(
                (
                    counts_array
                    * log_weighted_sum_probs(
                        self.pi[k][0],
                        logP[:, k + self.LLHOOD_TYPE_START_INDEX],
                        self.pi[k][1],
                        logP[:, self.MISSING_INDEX - 1],
                        self.pi[k][2],
                        logP[:, self.ANOMALIES_INDEX - 1],
                    )
                ).sum()
            )
        temp = normalize_log_probs(q)[y_i]
        if temp == 0:
            error = +800.0 / len(counts_array)
        else:
            error = -np.log(temp) / len(counts_array)

        return error

    def f_cols(self, w_j_z):
        # f: the objective function to minimize. (it is equal to - \sum_{all columns} log p(t=k|X) where k is the correct column type.)
        # Set params: init-transition-final
        runner, temp_w_j_z = self.current_runner.set_all_probabilities_z(w_j_z)

        # Generate probabilities
        self.all_probs = runner.generate_machine_probabilities(self.unique_vals)

        error = 0.0
        for i, (data_frame, labels) in enumerate(zip(self.training_params.dfs, self.training_params.labels)):
            for j, column_name in enumerate(list(data_frame.columns)):
                error += self.f_col(str(i), column_name, labels[j] - 1)
        return error

    def g_col_marginals(self, runner, i_, column_name, y_i):
        [temp_x, counts_array] = self.dfs_unique_vals_counts[i_][column_name]
        logP = np.array([self.all_probs[str(x_i)] for x_i in temp_x])

        # calculates posterior values of types
        r = []
        for k in range(self.K):
            r.append(
                (
                    counts_array
                    * log_weighted_sum_probs(
                        self.pi[k][0],
                        logP[:, k + self.LLHOOD_TYPE_START_INDEX],
                        self.pi[k][1],
                        logP[:, self.MISSING_INDEX - 1],
                        self.pi[k][2],
                        logP[:, self.ANOMALIES_INDEX - 1],
                    )
                ).sum()
            )

        # calculates the gradients for initial, transition, and final probabilities. (note that it is only for non-zero probabilities at the moment.)
        g_j = []
        for t in range(len(self.types)):
            x_i_indices = np.where(logP[:, t + 2] != LOG_EPS)[0]

            possible_states = [
                state
                for state in runner.machines[2 + t].states
                if runner.machines[2 + t].I[state] != LOG_EPS
            ]
            A = log_weighted_sum_probs(
                self.pi[t][0],
                logP[:, t + self.LLHOOD_TYPE_START_INDEX],
                self.pi[t][1],
                logP[:, self.MISSING_INDEX - 1],
                self.pi[t][2],
                logP[:, self.ANOMALIES_INDEX - 1],
            )
            temp_gra = np.exp(self.pi[t][0] + logP[:, t + 2] - A)

            # gradient for initial state parameters
            temp_g_j = []
            for state in possible_states:
                temp_g_j.append(
                    self.gradient_initial_optimized_new(
                        runner,
                        state,
                        t,
                        temp_x[x_i_indices],
                        r,
                        temp_gra[x_i_indices],
                        counts_array[x_i_indices],
                        y_i,
                    )
                )
            g_j = g_j + temp_g_j

            # gradient for transition parameters
            if t == 1:
                marginals = {
                    str(x_i): np.ones((len(x_i), 1, 1))
                    if p_x_i[t + 2] != LOG_EPS
                    else np.zeros((len(x_i), 1, 1))
                    for x_i, p_x_i in zip(temp_x, logP)
                }
            else:
                marginals = {
                    str(x_i): runner.machines[2 + t].run_forward_backward(str(x_i))
                    if p_x_i[t + 2] != LOG_EPS
                    else np.zeros((len(x_i), len(x_i)))
                    for x_i, p_x_i in zip(temp_x, logP)
                }
            state_indices = {}
            counter = 0
            temp_g_j = []
            for a in runner.machines[2 + t].T:
                for b in runner.machines[2 + t].T[a]:
                    for c in runner.machines[2 + t].T[a][b]:
                        state_indices[str(a) + "*" + str(b) + "*" + str(c)] = counter
                        temp_g_j.append(0)
                        counter += 1

            for x_i_index, (x_i, temp_gra_i, counts_array_i) in enumerate(
                zip(
                    temp_x[x_i_indices],
                    temp_gra[x_i_indices],
                    counts_array[x_i_indices],
                )
            ):
                if logP[x_i_index, t + 2] != LOG_EPS:
                    if t == 1:
                        common_chars = [x for x in runner.machines[t + 2].alphabet if x in list(str(x_i))]
                        for common_char in common_chars:
                            common_char_ls = np.where(list(str(x_i)) == common_char)[0]
                            for l in common_char_ls:
                                indices_nonzero = np.where(
                                    marginals[str(x_i)][l] != 0.0
                                )
                                if len(indices_nonzero[0]) != 0:
                                    q_s = indices_nonzero[0]
                                    q_primes = indices_nonzero[1]
                                    for q, q_prime in zip(q_s, q_primes):
                                        temp_g_j[
                                            state_indices[
                                                str(runner.machines[t + 2].states[q])
                                                + "*"
                                                + str(common_char)
                                                + "*"
                                                + str(
                                                    runner.machines[t + 2].states[
                                                        q_prime
                                                    ]
                                                )
                                            ]
                                        ] += self.gradient_transition_optimized_new_marginals(
                                            runner,
                                            marginals,
                                            runner.machines[t + 2].states[q],
                                            common_char,
                                            runner.machines[t + 2].states[q_prime],
                                            t,
                                            r,
                                            str(x_i),
                                            y_i,
                                            temp_gra_i,
                                            counts_array_i,
                                        )

                    else:
                        for l, alpha in enumerate(str(x_i)):
                            if alpha in runner.machines[t + 2].alphabet:
                                indices_nonzero = np.where(
                                    marginals[str(x_i)][l] != 0.0
                                )
                                if len(indices_nonzero[0]) != 0:
                                    q_s = indices_nonzero[0]
                                    q_primes = indices_nonzero[1]
                                    for q, q_prime in zip(q_s, q_primes):
                                        temp_g_j[
                                            state_indices[
                                                str(runner.machines[t + 2].states[q])
                                                + "*"
                                                + str(alpha)
                                                + "*"
                                                + str(
                                                    runner.machines[t + 2].states[
                                                        q_prime
                                                    ]
                                                )
                                            ]
                                        ] += self.gradient_transition_optimized_new_marginals(
                                            runner,
                                            marginals,
                                            runner.machines[t + 2].states[q],
                                            alpha,
                                            runner.machines[t + 2].states[q_prime],
                                            t,
                                            r,
                                            str(x_i),
                                            y_i,
                                            temp_gra_i,
                                            counts_array_i,
                                        )
            # print('transition done')
            g_j = g_j + temp_g_j

            # gradient for final-state parameters
            for state in runner.machines[2 + t].F:
                if runner.machines[2 + t].F[state] != LOG_EPS:
                    g_j.append(
                        self.gradient_final_optimized_new(
                            runner,
                            state,
                            t,
                            temp_x[x_i_indices],
                            r,
                            temp_gra[x_i_indices],
                            counts_array[x_i_indices],
                            y_i,
                        )
                    )
            # print('final done')

        return -np.array(g_j) / counts_array.sum()

    def g_cols(self, w_j_z):
        # calculates the gradient vector, i.e. df/dw (=df/dz * dz/dw) where f is the object function to minimize.
        # it returns -g_j because of minimizing instead of maximizing. see the objective function.

        # updates the parameters
        runner, temp_w_j_z = self.current_runner.set_all_probabilities_z(w_j_z)

        # generates probabilities
        self.all_probs = runner.generate_machine_probabilities(self.unique_vals)

        q_total = None
        counter_ = 0

        for i, (df, labels) in enumerate(zip(self.training_params.dfs, self.training_params.labels)):
            for j, column_name in enumerate(list(df.columns)):
                if counter_ == 0:
                    q_total = self.g_col_marginals(
                        runner, str(i), column_name, labels[j] - 1
                    )
                    counter_ += 1
                else:
                    q_total += self.g_col_marginals(
                        runner, str(i), column_name, labels[j] - 1
                    )

        return q_total

    ###################### GRADIENT HELPERS #######################
    def scale_wrt_type(self, gradient, q, t, y_i):
        temp = normalize_log_probs(q)[t]
        return gradient * (1 - temp) if t == y_i else -1 * gradient * temp

    def gradient_initial_optimized_new(
        self, runner, state, t, x, q, temp, counter, y_i
    ):
        exp_param = 1 - np.exp(runner.machines[2 + t].I[state])

        cs_temp = [
            runner.machines[2 + t].calculate_gradient_initial_state_optimized(
                str(x_i), state
            )
            for x_i in x
        ]
        cs = np.array(cs_temp)

        gradient = (temp * counter * cs * exp_param).sum()
        return self.scale_wrt_type(gradient, q, t, y_i)

    def gradient_transition_optimized_new_marginals(
        self, runner, marginals, a, b, c, t, q, x, y_i, temp_gra, counts_array
    ):

        temp_mult = (
            temp_gra
            * runner.machines[2 + t].calculate_gradient_abc_new_optimized_marginals(
                marginals[str(x)], str(x), a, b, c
            )
            * counts_array
        )
        exp_param = 1 - np.exp(runner.machines[2 + t].T[a][b][c])
        gradient = exp_param * temp_mult

        return self.scale_wrt_type(gradient, q, t, y_i)

    def gradient_final_optimized_new(
        self, runner, final_state, t, x, q, temp, counter, y_i
    ):
        exp_param = 1 - np.exp(runner.machines[2 + t].F[final_state])

        cs_temp = [
            runner.machines[2 + t].calculate_gradient_final_state_optimized(
                str(x_i), final_state
            )
            for x_i in x
        ]
        cs = np.array(cs_temp)
        gradient = sum(temp * counter * cs * exp_param)

        return self.scale_wrt_type(gradient, q, t, y_i)

    def get_all_parameters_z(self, runner):
        w_j = []
        for t in range(len(self.types)):
            for state in runner.machines[2 + t].I:
                if runner.machines[2 + t].I[state] != LOG_EPS:
                    w_j.append(runner.machines[2 + t].I_z[state])

            for a in runner.machines[2 + t].T_z:
                for b in runner.machines[2 + t].T[a]:
                    for c in runner.machines[2 + t].T[a][b]:
                        w_j.append(runner.machines[2 + t].T_z[a][b][c])

            for state in runner.machines[2 + t].F:
                if runner.machines[2 + t].F[state] != LOG_EPS:
                    w_j.append(runner.machines[2 + t].F_z[state])

        return w_j

    @staticmethod
    def normalize_a_state(F, T, a):
        # find maximum log probability
        params = []
        for b in T[a]:
            for c in T[a][b]:
                params.append(T[a][b][c])

        if F[a] != LOG_EPS:
            params.append(F[a])

        log_mx = max(params)
        sm = sum([np.exp(param - log_mx) for param in params])

        # normalize
        for b in T[a]:
            for c in T[a][b]:
                T[a][b][c] = np.log(np.exp(T[a][b][c] - log_mx) / sm)
        if F[a] != LOG_EPS:
            if log_mx == LOG_EPS:
                F[a] = 0.0
            else:
                F[a] = np.log(np.exp(F[a] - log_mx) / sm)

        return F, T

    @staticmethod
    def normalize_initial(I):
        # find maximum log probability
        log_mx = LOG_EPS
        for a in I:
            if I[a] != LOG_EPS and I[a] > log_mx:
                log_mx = I[a]
        # sum
        sm = 0
        for a in I:
            if I[a] != LOG_EPS:
                sm += np.exp(I[a] - log_mx)

        # normalize
        for a in I:
            if I[a] != LOG_EPS:
                I[a] = I[a] - log_mx - np.log(sm)

        return I

    @staticmethod
    def normalize_final(F, T):
        for state in F:
            F, T = Model.normalize_a_state(F, T, state)

        return F, T
