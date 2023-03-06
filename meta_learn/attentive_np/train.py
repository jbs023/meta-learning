import argparse
import os
import gpytorch
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from meta_learn.attentive_np.model import ANP

from meta_learn.cnp.model import CNP
from meta_learn.sine_dataset import Task_Distribution
from torch.distributions import Normal

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def main(config):
    path = config.path
    n_shot_train = 10
    n_shot_test = int(n_shot_train/2)
    train_range = (-5.0, 5.0)
    test_range = (-5.0, 5.0)  # This must be (-5, +10) for the out-of-range condition
    logdir = "{}/{}/{}_shot".format(path, "cnp", n_shot_train)

    #Set up datasets
    train_task = Task_Distribution(
        amplitude_min=0.1,
        amplitude_max=5.0,
        phase_min=0.0,
        phase_max=np.pi,
        x_min=train_range[0],
        x_max=train_range[1],
        family="sine",
    )
    test_task = Task_Distribution(
        amplitude_min=0.1,
        amplitude_max=5.0,
        phase_min=0.0,
        phase_max=np.pi,
        x_min=test_range[0],
        x_max=test_range[1],
        family="sine",
    )

    model = ANP(2, 64)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device=device)
    criterion = nn.MSELoss()

    ## Training
    tot_iterations = 10000
    mse_list = list()
    loss_list = list()
    for epoch in range(tot_iterations):
        #Zero out the gradient
        optimizer.zero_grad()

        #Sampling from the tasks
        x_all, y_all = train_task.sample_task().sample_data(n_shot_train, noise=0.1)
        indices = np.arange(n_shot_train)
        np.random.shuffle(indices)
        support_indices = np.sort(indices[0:n_shot_test])

        query_indices = np.sort(indices[n_shot_test:])
        x_support = x_all[support_indices]
        y_support = y_all[support_indices]
        x_query = x_all[query_indices]
        y_query = y_all[query_indices]

        mu, sigma, loss = model(x_support, y_support, x_query, y_query)
        loss.backward()
        optimizer.step()

        mse = criterion(mu.squeeze(), y_query.squeeze())
        loss_list.append(loss.item())
        mse_list.append(mse.item())

        # ---- print some stuff ----
        if epoch % 100 == 0:
            print(
                "[%d] - Loss: %.3f  MSE: %.3f"
                % (
                    epoch,
                    np.mean(loss_list),
                    np.mean(mse_list)
                )
            )
            mse_list = list()
            loss_list = list()

    ## Test phase on a new sine/cosine wave
    print("Test, please wait...")

    with torch.no_grad():
        tot_iterations = 500
        mse_list = list()
        sample_size = 10
        n_shot_test = 5
        for epoch in range(tot_iterations):
            x_all, y_all = test_task.sample_task().sample_data(sample_size, noise=0.1, sort=True)
            indices = np.arange(sample_size)
            np.random.shuffle(indices)
            support_indices = np.sort(indices[0:n_shot_test])

            query_indices = np.sort(indices[n_shot_test:])
            x_support = x_all[support_indices]
            y_support = y_all[support_indices]
            x_query = x_all[query_indices]
            y_query = y_all[query_indices]

            mu, sigma, loss = model(x_support, y_support, x_query, y_query)
            optimizer.step()

            mse = criterion(mu.squeeze(), y_query.squeeze())
            loss_list.append(loss.item())
            mse_list.append(mse.item())

        print("-------------------")
        print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
        print("-------------------")

        #Plotting
        for i in range(10):
            sample_task = test_task.sample_task()
            x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
            query_indices = np.sort(indices[n_shot_test:])
            x_support = x_all[support_indices]
            y_support = y_all[support_indices]
            x_query = x_all[query_indices]
            y_query = y_all[query_indices]

            mu, sigma, loss = model(x_support, y_support, x_query, y_query)
            optimizer.step()

            # Evaluation on all data
            mean = np.squeeze(mu)
            lower = np.squeeze(mu - sigma)
            upper = np.squeeze(mu + sigma)

            # Plot
            fig, ax = plt.subplots()

            # true-curve
            true_curve = np.linspace(train_range[0], train_range[1], 1000)
            true_curve = [sample_task.true_function(x) for x in true_curve]
            ax.plot(
                np.linspace(train_range[0], train_range[1], 1000),
                true_curve,
                color="blue",
                linewidth=2.0,
            )
            if train_range[1] < test_range[1]:
                dotted_curve = np.linspace(train_range[1], test_range[1], 1000)
                dotted_curve = [sample_task.true_function(x) for x in dotted_curve]
                ax.plot(
                    np.linspace(train_range[1], test_range[1], 1000),
                    dotted_curve,
                    color="blue",
                    linestyle="--",
                    linewidth=2.0,
                )

            # query points (ground-truth)
            ax.scatter(x_query, y_query, color='red')
            
            # query points (predicted)
            ax.plot(np.squeeze(x_query), mean.detach().numpy(), color="red", linewidth=2.0)
            # print(lower.detach().numpy(), upper.detach().numpy())
            # print()
            ax.fill_between(
                np.squeeze(x_query),
                lower.detach().numpy(),
                upper.detach().numpy(),
                alpha=0.1,
                color="red",
            )
            # support points
            ax.scatter(x_support, y_support, color="darkblue", marker="*", s=50, zorder=10)

            # all points
            # ax.scatter(x_all.numpy(), y_all.numpy())
            # plt.show()
            plt.ylim(-6.0, 6.0)
            plt.xlim(test_range[0], test_range[1])
            plt.savefig("plot_ANP_" + str(i) + ".png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=os.getcwd())
    main(parser.parse_args())
