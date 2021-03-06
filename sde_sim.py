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

	# evaluate "midpoint hand sum" of W_{(t_j + t_j+1)/2} * (W_{t_j+1} - W_{t_j}) = 0.5 * W(T)^2
	# where W_{(t_j + t_j+1)/2} = (W_{t_j} + W_{t_j+1})/2 + N(0, dt/4)
	stratonovich = sum((0.5 * (W[0:-1] + W[1:]) + \
						0.5 * np.sqrt(dt) * np.random.normal(0, 1, N - 1)) * dW[1:])
	stratonovich_error = np.abs(stratonovich - 0.5 * W[-1]**2)

	print("Ito: ", ito, " Stantanovich: ", stratonovich)
	print("Ito Error: ", ito_error, " Stantanovich Error: ", stratonovich_error)


def euler_maruyama(N, T, X0, LAMBDA, MU, R):
	# solves SDE dX = lambda * X * dt + mu * X * dW, X(0) = X0
	# the exact solution is X(t) = X(0) * exp((lambda - 0.5 * mu^2)t + mu * W(t))
	dt = T / float(N)
	dW = np.sqrt(dt) * np.random.normal(0, 1, N)
	dW[0] = 0
	W = np.cumsum(dW)
	t = np.arange(0.0, T, dt)

	X_true = X0 * np.exp((LAMBDA - 0.5 * MU**2) * t + MU * W)

	Dt = R * dt
	L = int(N / R)
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


def euler_maruyama_strong_convergence(M, N, T, X0, LAMBDA, MU):
	# solves SDE dX = lambda * X * dt + mu * X * dW, X(0) = X0
	# and examines strong convergence E|X_L - X(T)| at T = 1
	X_err = np.zeros((M, 5))
	dt = T / float(N)

	for s in range(M):
		dW = np.sqrt(dt) * np.random.normal(0, 1, N)
		dW[0] = 0
		W = np.cumsum(dW)
		X_true = X0 * np.exp((LAMBDA - 0.5 * MU**2) * 1 + MU * W[-1])

		for p in range(5):
			R = 2**(p - 1)
			Dt = R * dt
			L = int(N / R)
			X_temp = X0

			for l in range(1, L):
				W_inc = np.sum(dW[R * (l - 1) + 1 : R * l + 1])
				X_temp = X_temp + LAMBDA * X_temp * Dt + MU * X_temp * W_inc

			X_err[s, p] = np.abs(X_temp - X_true)

	Dt = dt * 2**np.arange(5)
	X_err_mean = np.mean(X_err, axis=0)
	A = np.ones((5, 2))
	A[:, 1] = np.log(Dt)
	rhs = np.log(X_err_mean)
	sol = np.linalg.lstsq(A, rhs)
	print("Power: ", sol[0][1])

	plt.loglog(Dt, X_err_mean)
	plt.loglog(Dt, Dt**0.5)
	plt.xlabel("Delta t")
	plt.ylabel("Sample average of |X(T) - X_L|")
	plt.show()


def main():
	M = 1000
	N = 512
	T = 1
	brownian_path(N, T)
	f_brownian_path(M, N, T)
	stochastic_integral(N, T)
	euler_maruyama(N, T, X0=1, LAMBDA=2, MU=1, R=2)
	euler_maruyama_strong_convergence(M, N, T, X0=1, LAMBDA=2, MU=1)


if __name__ == '__main__':
	main()
