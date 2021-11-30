import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs experiments for the swarm consensus simulation.')
    parser.add_argument('-d', '--dirpath', default="./", required=True, 
                        help='Path to directory with csv file')

    args = parser.parse_args()
    fig = plt.figure(figsize=(10, 10))
    files = [f for f in os.listdir(args.dirpath) if os.path.isfile(f)]
    for f in files:
        if ".csv" in f:
            df = pd.read_csv(f)
            plt.plot(df.index, df["Accuracy"], label=f)
            # plt.plot(df.index, df["Loss"], label=f)

    plt.title("Training")
    plt.xlabel("Epochs")
    plt.ylabel("BCELoss")

    plt.legend()
    plt.show()