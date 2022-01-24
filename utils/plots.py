
import matplotlib.pyplot as plt

def plot_accuracy_and_loss(epochs, accs, losses):
    fig, ax1 = plt.subplots()

    del epochs[0]
    del accs[0]
    del losses[0]

    color = 'tab:red'
    ax1.set_xlabel('Iterations (n)')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, accs , color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()