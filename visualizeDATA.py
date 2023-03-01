import matplotlib.pyplot as plt


def plot(x, y, correct_prefixes):
    plt.plot(x,y)

    plt.xlabel('MAC Address')
    plt.ylabel('Prediction')

    for prefix in correct_prefixes:
        plt.axvspan(prefix, prefix+0x1_000000, color='green', alpha=0.5)
    
    plt.show()

