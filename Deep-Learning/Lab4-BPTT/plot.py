import matplotlib.pyplot as plt
import numpy as np

# plot
def show_result(title, x1, x2, xd, y1, y2, yd, y):
    plt.figure()
    plt.title(title, fontsize=18)
    plt.xlim(x1-xd/4, x2+xd/4)
    plt.xticks(np.arange(x1, x2+xd/4, xd))
    plt.ylim(y1-yd/4, y2+yd/4)
    plt.yticks(np.arange(y1, y2+yd/4, yd))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    x = [i for i in range(1, x2 + 1)]
    plt.plot(x, y, color='teal', marker='o', linewidth=1.0)

    plt.grid()
    plt.show()

