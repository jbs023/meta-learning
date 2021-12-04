import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

label_dict = {
    "siamese_final_loss_cnn_30k_v2.csv" : "CNN",
    "siamese_final_loss_cnn_d_30k_v2.csv" : "CNN + Dist",
    "siamese_final_loss_ff_30k_v2.csv" : "FFNN",
    "siamese_final_loss_ff_dist_30k_v2.csv" : "FFNN + Dist",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs experiments for the swarm consensus simulation.')
    parser.add_argument('-d', '--dirpath', default="./", required=True, 
                        help='Path to directory with csv file')

    args = parser.parse_args()
    full_dir_path = os.path.abspath(args.dirpath)
    fig = plt.figure(figsize=(10, 10))
    files = [f for f in os.listdir(full_dir_path)]
    for f in files:
        if ".csv" in f:
            df = pd.read_csv(os.path.join(full_dir_path, f))
            plt.plot(df.index, df["Accuracy"], label=f)
            # plt.plot(df.index, df["Loss"], label=f)

    plt.title("Training")
    plt.xlabel("Epochs")
    plt.ylabel("BCELoss")

    plt.legend()
    plt.show()