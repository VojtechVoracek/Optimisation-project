
import matplotlib.pyplot as plt
from optimized_functions import *
from main import *

plt.rcParams['text.usetex'] = True


def plot_runs_sphere(dimensions, num_of_epochs, num_of_runs, tol=1e-7, plot_average=False, q=None):
    """
            Create graphs capturing the performance of SGD and RR algorithms on the sphere function
        :param dimensions: (d, ) int
                Array of dimensions of the sphere function.
        :param num_of_epochs: int
                Maximal number of epochs for both algorithms.
        :param num_of_runs: int
                Number of independent runs of both algorithms.
        :param tol: float
                Tolerance, epsilon.
        :param plot_average:
                If True: plot the distance of q-suffix average \overline{x}_{q,k} to the optimal solution x^*
                If False plot the distance of the current x to the optimal solution x^*
        :param q:
                The q-suffix parameter. Define the number of last iterates from which the average is computed.
        :return: None
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axs = [ax1, ax2, ax3, ax4]

    for d in range(len(dimensions)):

        dimension = dimensions[d]
        optimized_function = SphereFunction(dimension)

        scg_average = np.zeros(num_of_epochs + 1)
        rr_average = np.zeros(num_of_epochs + 1)

        for i in range(num_of_runs):

            print(i)
            x_0 = np.random.uniform(low=-10, high=10, size=dimension)

            scg_xs, _ = SGD(x_0, num_of_epochs, tol, optimized_function)
            rr_xs, _ = random_reshuffling(x_0, num_of_epochs, tol, optimized_function)

            if plot_average:
                # Change iterates to the q-suffix average

                # q * k
                sgd_size = int(q * (len(scg_xs) + 1))
                rr_size = int(q * (len(rr_xs) + 1))

                if len(scg_xs) < num_of_epochs + 1:
                    diff = min(num_of_epochs + 1 - len(scg_xs), sgd_size)
                    scg_xs = np.append(scg_xs, np.ones((diff, 1)) * optimized_function.x_star, axis=0)

                if len(rr_xs) < num_of_epochs + 1:
                    diff = min(num_of_epochs + 1 - len(rr_xs), rr_size)
                    rr_xs = np.append(rr_xs, np.ones((diff, 1)) * optimized_function.x_star, axis=0)

                for i in range(len(scg_xs), 0, -1):
                    if i >= sgd_size:
                        scg_xs[i - 1, :] = np.sum(scg_xs[(i - sgd_size):i, :], axis=0) / sgd_size
                    else:
                        scg_xs[i - 1, :] = np.sum(scg_xs[:i], axis=0) / i

                for i in range(len(rr_xs), 0, -1):
                    if i >= rr_size:
                        rr_xs[i - 1, :] = np.sum(rr_xs[(i - rr_size):i, :], axis=0) / rr_size
                    else:
                        rr_xs[i - 1, :] = np.sum(rr_xs[:i], axis=0) / i

            # calculate distance to the optimal solution
            scg_xs = np.linalg.norm(scg_xs - optimized_function.x_star, axis=1)
            rr_xs = np.linalg.norm(rr_xs - optimized_function.x_star, axis=1)

            scg_xs = scg_xs[np.where(scg_xs >= tol)]
            rr_xs = rr_xs[np.where(rr_xs >= tol)]

            # plot individual runs
            axs[d].plot(scg_xs, linewidth=0.5, color="dodgerblue")
            axs[d].plot(rr_xs, linewidth=0.5, color="lightcoral")

            # add this run to average over all runs
            scg_average[:len(scg_xs)] = scg_average[:len(scg_xs)] + scg_xs
            rr_average[:len(rr_xs)] = rr_average[:len(rr_xs)] + rr_xs

        scg_average = scg_average / num_of_runs
        rr_average = rr_average / num_of_runs

        scg_average = scg_average[np.where(scg_average >= tol)]
        rr_average = rr_average[np.where(rr_average >= tol)]

        axs[d].set_yscale('log')
        if d % 2 == 1:
            axs[d].yaxis.tick_right()
        if d >= 2:
            axs[d].set_xlabel("Epoch")

        if plot_average:
            axs[d].set_ylabel(r'$||\overline{x}_{q,k} - x^{*}||$')
        else:
            axs[d].set_ylabel(r'$||x - x^{*}||$')

        # make the O(1 / k^s) ratio curve
        k = np.arange(len(scg_average))
        ks = 1 / (k + 1) ** 0.6

        axs[d].legend([axs[d].plot(scg_average, color="blue")[0], axs[d].plot(rr_average, color="red")[0],
                       axs[d].plot(k, ks, color="black", linestyle="--")[0]],
                      ["SGD", "RR", r'$\mathcal{O}(1 / k^{s})$'])

        subtitle = "d=" + str(dimension)

        axs[d].set_title(subtitle, y=0.8, x=0.42)

    fig.suptitle("Sphere function", size=20, y=0.95)

    if plot_average:
        fig.savefig("Sphere_runs_average.png")
    else:
        fig.savefig("Sphere_runs.png")


def plot_runs_component_func(num_of_epochs, num_of_runs, tol=-1e-7, plot_average=False, q=None):
    """
            Create graphs capturing the performance of SGD and RR algorithms on the sphere function
        :param num_of_epochs: int
                Maximal number of epochs for both algorithms.
        :param num_of_runs: int
                Number of independent runs of both algorithms.
        :param tol: float
                Tolerance, epsilon.
        :param plot_average:
                If True: plot the distance of q-suffix average \overline{x}_{q,k} to the optimal solution x^*
                If False plot the distance of the current x to the optimal solution x^*
        :param q:
                The q-suffix parameter. Define the number of last iterates from which the average is computed.
        :return: None
    """
    fig = plt.figure()
    optimized_function = ComponentFunction()

    scg_average = np.zeros(num_of_epochs + 1)
    rr_average = np.zeros(num_of_epochs + 1)

    for i in range(num_of_runs):
        print(i)
        x_0 = np.random.uniform(low=-1, high=1, size=1)
        scg_xs, scg_objectives = SGD(x_0, num_of_epochs, tol, optimized_function)
        rr_xs, rr_objectives = random_reshuffling(x_0, num_of_epochs, tol, optimized_function)

        if plot_average:
            # Change iterates to the q-suffix average

            # q * k
            sgd_size = int(q * (len(scg_xs) + 1))
            rr_size = int(q * (len(rr_xs) + 1))

            if len(scg_xs) < num_of_epochs + 1:
                diff = min(num_of_epochs + 1 - len(scg_xs), sgd_size)
                scg_xs = np.append(scg_xs, np.ones((diff, 1)) * optimized_function.x_star, axis=0)

            if len(rr_xs) < num_of_epochs + 1:
                diff = min(num_of_epochs + 1 - len(rr_xs), rr_size)
                rr_xs = np.append(rr_xs, np.ones((diff, 1)) * optimized_function.x_star, axis=0)

            for i in range(len(scg_xs), 0, -1):
                if i >= sgd_size:
                    scg_xs[i - 1, :] = np.sum(scg_xs[(i - sgd_size):i, :], axis=0) / sgd_size
                else:
                    scg_xs[i - 1, :] = np.sum(scg_xs[:i], axis=0) / i

            for i in range(len(rr_xs), 0, -1):
                if i >= rr_size:
                    rr_xs[i - 1, :] = np.sum(rr_xs[(i - rr_size):i, :], axis=0) / rr_size
                else:
                    rr_xs[i - 1, :] = np.sum(rr_xs[:i], axis=0) / i

        # calculate distance to the optimal solution
        scg_xs = np.linalg.norm(scg_xs - optimized_function.x_star, axis=1)
        rr_xs = np.linalg.norm(rr_xs - optimized_function.x_star, axis=1)

        scg_xs = scg_xs[np.where(scg_xs >= tol)]
        rr_xs = rr_xs[np.where(rr_xs >= tol)]

        # plot individual runs
        plt.plot(scg_xs, linewidth=0.1, color="dodgerblue")
        plt.plot(rr_xs, linewidth=0.1, color="lightcoral")

        # add this run to average over all runs
        scg_average[:len(scg_xs)] = scg_average[:len(scg_xs)] + scg_xs
        rr_average[:len(rr_xs)] = rr_average[:len(rr_xs)] + rr_xs

    scg_average = scg_average / num_of_runs
    rr_average = rr_average / num_of_runs

    scg_average = scg_average[np.where(scg_average >= tol)]
    rr_average = rr_average[np.where(rr_average >= tol)]

    plt.yscale('log')
    plt.xlabel("Epoch")

    plt.ylabel(r'$||\overline{x}_{q,k} - x^{*}||$')

    # make the O(1 / k^s) ratio curve
    k = np.arange(len(scg_average))
    ks = 1 / (k + 1) ** 0.9

    plt.legend([plt.plot(scg_average, color="blue")[0], plt.plot(rr_average, color="red")[0],
                plt.plot(k, ks, color="black", linestyle="--")[0]], ["SGD", "RR", r'$\mathcal{O}(1 / k^{s})$'])

    fig.suptitle("Component function", size=20, y=0.95)
    fig.savefig("component_runs.png")


def plot_runs_lin_reg(num_of_epochs, num_of_runs, A, b, tol=-1e-7, plot_average=False, q=None):
    """
            Create graphs capturing the performance of SGD and RR algorithms on the sphere function
        :param num_of_epochs: int
                Maximal number of epochs for both algorithms.
        :param num_of_runs: int
                Number of independent runs of both algorithms.
        :param A : [num_of_functions, d] float
                The data matrix for linear regression.
        :param b : (num_of_functions,) float
                The target vector for linear regression.
        :param tol: float
                Tolerance, epsilon.
        :param plot_average:
                If True: plot the distance of q-suffix average \overline{x}_{q,k} to the optimal solution x^*
                If False plot the distance of the current x to the optimal solution x^*
        :param q:
                The q-suffix parameter. Define the number of last iterates from which the average is computed.
        :return: None
    """
    fig = plt.figure()
    optimized_function = LinearRegression(A, b)

    scg_average = np.zeros(num_of_epochs * A.shape[0] + 1)
    rr_average = np.zeros(num_of_epochs * A.shape[0] + 1)

    for i in range(num_of_runs):
        print(i)
        x_0 = np.random.uniform(low=-10, high=10, size=A.shape[1])
        scg_xs, scg_objectives = SGD(x_0, num_of_epochs, tol, optimized_function)
        rr_xs, rr_objectives = random_reshuffling(x_0, num_of_epochs, tol, optimized_function)

        if plot_average:
            # Change iterates to the q-suffix average

            # q * k
            sgd_size = int(q * (len(scg_xs) + 1))
            rr_size = int(q * (len(rr_xs) + 1))

            if len(scg_xs) < num_of_epochs + 1:
                print("We")
                diff = min(num_of_epochs + 1 - len(scg_xs), sgd_size)
                scg_xs = np.append(scg_xs, np.ones((diff, 1)) * optimized_function.x_star, axis=0)

            if len(rr_xs) < num_of_epochs + 1:
                diff = min(num_of_epochs + 1 - len(rr_xs), rr_size)
                rr_xs = np.append(rr_xs, np.ones((diff, 1)) * optimized_function.x_star, axis=0)

            for i in range(len(scg_xs), 0, -1):
                if i >= sgd_size:
                    scg_xs[i - 1, :] = np.sum(scg_xs[(i - sgd_size):i, :], axis=0) / sgd_size
                else:
                    scg_xs[i - 1, :] = np.sum(scg_xs[:i], axis=0) / i

            for i in range(len(rr_xs), 0, -1):
                if i >= rr_size:
                    rr_xs[i - 1, :] = np.sum(rr_xs[(i - rr_size):i, :], axis=0) / rr_size
                else:
                    rr_xs[i - 1, :] = np.sum(rr_xs[:i], axis=0) / i

        # calculate distance to the optimal solution
        scg_xs = np.linalg.norm(scg_xs - optimized_function.x_star, axis=1)
        rr_xs = np.linalg.norm(rr_xs - optimized_function.x_star, axis=1)

        scg_xs = scg_xs[np.where(scg_xs >= tol)]
        rr_xs = rr_xs[np.where(rr_xs >= tol)]

        # plot individual runs
        plt.plot(scg_xs, linewidth=0.1, color="dodgerblue")
        plt.plot(rr_xs, linewidth=0.1, color="lightcoral")

        # add this run to average over all runs
        scg_average[:len(scg_xs)] = scg_average[:len(scg_xs)] + scg_xs
        rr_average[:len(rr_xs)] = rr_average[:len(rr_xs)] + rr_xs

    scg_average = scg_average / num_of_runs
    rr_average = rr_average / num_of_runs

    scg_average = scg_average[np.where(scg_average >= tol)]
    rr_average = rr_average[np.where(rr_average >= tol)]

    plt.yscale('log')
    plt.xlabel("Epoch")

    plt.ylabel(r'$||\overline{x}_{q,k} - x^{*}||$')

    # make the O(1 / k^s) ratio curve
    k = np.arange(len(scg_average))
    ks = 10000 / (k + 1) ** 0.9

    plt.legend([plt.plot(scg_average, color="blue")[0], plt.plot(rr_average, color="red")[0],
                plt.plot(k, ks, color="black", linestyle="--")[0]], ["SGD", "RR", r'$\mathcal{O}(1 / k^{s})$'])

    fig.suptitle("Linear regression - diabetes dataset", size=20, y=0.95)
    fig.savefig("regression_runs_a.png")
