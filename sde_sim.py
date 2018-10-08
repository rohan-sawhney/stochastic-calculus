import numpy as np
import matplotlib.pyplot as plt


def brownian_path(N, T):
	dt = T / float(N)
	dW = np.sqrt(dt) * np.random.normal(0, 1, N)
	dW[0] = 0
	W = np.cumsum(dW)

	plt.plot(np.arange(0.0, T, dt), W)
	plt.xlabel('t')
	plt.ylabel('W(t)')
	plt.show()


def f_brownian_path(M, N, T):
	dt = T / float(N)
	dW = np.sqrt(dt) * np.random.normal(0, 1, (M, N))
	dW[:, 0] = 0
	W = np.cumsum(dW, axis=1)
	t = np.arange(0.0, T, dt)

	# define function
	U = np.exp(np.tile(t, [M, 1]) + 0.5 * W)
	U_mean = np.mean(U, axis=0)

	for i in range(M/10):
		plt.plot(t, U[i, :], color='red')
	plt.plot(t, U_mean, color='blue', label="mean of %d paths" % M)
	plt.xlabel('t')
	plt.ylabel('U(t)')
	plt.legend()
	plt.show()

	print("Sample error: ", np.linalg.norm((U_mean - np.exp(9.0 * t / 8.0)), np.inf))


def stochastic_integral(N, T):
	dt = T / float(N)
	dW = np.sqrt(dt) * np.random.normal(0, 1, N)
	dW[0] = 0
	W = np.cumsum(dW)

	# evaluate "left hand sum" of W_{t_j} * (W_{t_j+1} - W_{t_j}) = 0.5 * (W(T)^2 - T)
	ito = np.sum(W[0:-1] * dW[1:])
	ito_error = np.abs(ito - 0.5 * (W[-1]**2 - T))

	# evaluate "midpoint hand sum" of W_{(t_j + t_j+1)/2} * (W_{t_j+1} - W_{t_j}) = 0.5 * (W(T)^2 - T)
	# where W_{(t_j + t_j+1)/2} = (W_{t_j} + W_{t_j+1})/2 + N(0, dt/4)
	stratonovich = sum((0.5 * (W[0:-1] + W[1:]) + \
						0.5 * np.sqrt(dt) * np.random.normal(0, 1, N - 1)) * dW[1:])
	stratonovich_error = np.abs(stratonovich - 0.5 * W[-1]**2)

	print("Ito: ", ito, " Stantanovich: ", stratonovich)
	print("Ito Error: ", ito_error, " Stantanovich Error: ", stratonovich_error)


def euler_maruyama(N, T, X0, LAMBDA, MU, R):
	# solves SDE dX = lambda * X dt + mu * X * dW, X(0) = X0
	# the exact solution is X(t) = X(0) exp((lambda - 0.5 * mu^2)t + mu * W(t))
	dt = T / float(N)
	dW = np.sqrt(dt) * np.random.normal(0, 1, N)
	dW[0] = 0
	W = np.cumsum(dW)
	t = np.arange(0.0, T, dt)

	X_true = X0 * np.exp((LAMBDA - 0.5 * MU**2) * t + MU * W)

	Dt = R * dt
	L = N / R
	X = np.zeros(L)
	X[0] = X0
	X_temp = X0
	for l in range(1, L):
		W_inc = np.sum(dW[R * (l - 1) + 1 : R * l + 1])
		X_temp = X_temp + LAMBDA * X_temp * Dt + MU * X_temp * W_inc
		X[l] = X_temp

	plt.plot(t, X_true, label="true")
	plt.plot(np.arange(0.0, T, Dt), X, label="sim")
	plt.xlabel("t")
	plt.ylabel("X(t)")
	plt.legend()
	plt.show()

	error = np.abs(X[-1] - X_true[-1])
	print("Error: ", error)


def euler_maruyama_strong_convergence():
	# TODO
	pass


def euler_maruyama_weak_convergence():
	# TODO
	pass


def milstein_strong_convergence():
	# TODO
	pass


def euler_maruyama_stability():
	# TODO
	pass


def stochastic_chain_rule():
	# TODO
	pass


def main():
	M = 1000
	N = 500
	T = 1
	brownian_path(N, T)
	f_brownian_path(M, N, T)
	stochastic_integral(N, T)
	euler_maruyama(N, T, X0=1, LAMBDA=2, MU=1, R=2)
	euler_maruyama_strong_convergence()
	euler_maruyama_weak_convergence()
	milstein_strong_convergence()
	euler_maruyama_stability()
	stochastic_chain_rule()


if __name__ == '__main__':
	main()
