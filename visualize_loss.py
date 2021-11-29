import matplotlib.pyplot as plt
import pandas as pd
import os

label_dict = {
    "siamese_final_loss_cnn_30k_v2.csv" : "CNN",
    "siamese_final_loss_cnn_d_30k_v2.csv" : "CNN + Dist",
    "siamese_final_loss_ff_30k_v2.csv" : "FFNN",
    "siamese_final_loss_ff_dist_30k_v2.csv" : "FFNN + Dist",
}

if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 10))
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    files = sorted(files)
    for f in files:
        if ".csv" in f:
            df = pd.read_csv(f)
            plt.plot(df.index, df["Loss"], label=label_dict[f])

    plt.title("Training")
    plt.xlabel("Epochs")
    plt.ylabel("BCELoss")

    plt.legend()
    plt.show()