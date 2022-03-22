import os
import matplotlib.pyplot as plt

PATH = ""  # path to .txt files with losses


def make_plot():
    all_lines = []
    for file in os.listdir(PATH):
        if file.endswith("loss.txt"):
            with open(PATH + "\\" + file, 'r') as f:
                lines = [line.rstrip() for line in f]
                all_lines.append(lines)

    x = [i for i in range(len(all_lines[0]))]

    final_y = [0]*len(all_lines[0])
    for ys in all_lines:
        for i, y in enumerate(ys):
            final_y[i] += float(y)

    l = len(all_lines)
    for i in range(len(final_y)):
        final_y[i] /= l

    plt.plot(x, final_y, label="Average loss, crop_size=128")
    plt.xlabel("Epochs")
    plt.ylabel("Loss, MSE")
    plt.legend()
    plt.savefig("loss_dip.svg", format="svg")


if __name__ == "__main__":
    make_plot()
