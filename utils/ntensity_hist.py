import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PATH = ""  # path to the image


def make_hist():
    try:
        img_pil = Image.open(PATH)
    except Exception as e:
        print(f"Failed to open the image {e}")
        assert False

    n_img = np.array(img_pil)

    plt.hist(n_img.flatten(), bins=64, density=True)
    plt.xlabel("Intensity level")
    plt.savefig("intensity.svg", format="svg")


if __name__ == "__main__":
    make_hist()
