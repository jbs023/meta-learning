import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("siamese_cnn_no_transform_loss.csv")
    # df = pd.read_csv("test.csv")
    x = df.index
    y = df["Loss"]
    plt.plot(x, y)

    plt.title("Training")
    plt.xlabel("Epochs")
    plt.ylabel("BCELoss")
    plt.show()