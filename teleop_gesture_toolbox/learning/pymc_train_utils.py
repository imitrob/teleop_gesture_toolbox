import matplotlib.pyplot as plt


def plot_hist(mu, sigma):
    plt.hist(mu, bins=100)
    plt.show()
    plt.hist(sigma, bins=100)
    plt.show()

    