#This code is a modified version of: https://github.com/BayesWatch/deep-kernel-transfer/blob/master/sines/train_DKT.py

import gpytorch

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from gpytorch.mlls import VariationalELBO
from meta_learn.dkt.model import ApproximateGPModel, FCResNet, initial_values
from meta_learn.sine_dataset import Task_Distribution

sns.set()

import matplotlib.pyplot as plt
import numpy as np

#TODO: Integrate this dataset into the CNP code? Maybe? It's easier to visualize.
def main():
    ## Defining model
    n_shot_train = 10
    n_shot_test = 5
    train_range = (-5.0, 5.0)
    test_range = (-5.0, 5.0)  # This must be (-5, +10) for the out-of-range condition
    criterion = nn.MSELoss()

    #Set up datasets
    task_train = Task_Distribution(
        amplitude_min=0.1,
        amplitude_max=5.0,
        phase_min=0.0,
        phase_max=np.pi,
        x_min=train_range[0],
        x_max=train_range[1],
        family="sine",
    )
    tasks_test = Task_Distribution(
        amplitude_min=0.1,
        amplitude_max=5.0,
        phase_min=0.0,
        phase_max=np.pi,
        x_min=test_range[0],
        x_max=test_range[1],
        family="sine",
    )


    net = FCResNet()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    dummy_inputs = torch.zeros([n_shot_train, 40])
    dummy_labels = torch.zeros([n_shot_train])

    init_inducing_points, init_lengthscale = initial_values(task_train, net, n_shot_train)
    gp = ApproximateGPModel(init_inducing_points, init_lengthscale, likelihood)
    elbo = VariationalELBO(likelihood, gp, num_data=n_shot_train)
    optimizer = torch.optim.Adam(
        [
            {"params": gp.parameters(), "lr": 1e-3},
            {"params": net.parameters(), "lr": 1e-3},
        ]
    )

    ## Training
    likelihood.train()
    gp.train()
    net.train()
    if torch.cuda.is_available():
        gp = gp.cuda()
        likelihood = likelihood.cuda()
        net = net.cuda()

    tot_iterations = 500  # 50000
    mse_list = list()
    loss_list = list()
    for epoch in range(tot_iterations):
        optimizer.zero_grad()
        inputs, labels = task_train.sample_task().sample_data(n_shot_train, noise=0.1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        z = net(inputs)
        # gp.set_train_data(inputs=z, targets=labels)
        print(z.shape, labels.shape)
        fantasy_gp = gp.get_fantasy_model(inputs=z, targets=labels)
        predictions = gp(z)
        loss = -elbo(predictions, labels)
        loss.backward()
        optimizer.step()
        mse = criterion(predictions.mean, labels)
        loss_list.append(loss.item())
        mse_list.append(mse.item())

        # ---- print some stuff ----
        if epoch % 100 == 0:
            print(
                "[%d] - Loss: %.3f  MSE: %.3f  lengthscale: %.3f   noise: %.3f"
                % (
                    epoch,
                    np.mean(loss_list),
                    np.mean(mse_list),
                    0.0,  # gp.covar_module.base_kernel.lengthscale.item(),
                    gp.likelihood.noise.item(),
                )
            )
            mse_list = list()
            loss_list = list()

    ## Test phase on a new sine/cosine wave
    print("Test, please wait...")

    likelihood.eval()
    net.eval()
    tot_iterations = 500
    mse_list = list()
    for epoch in range(tot_iterations):
        sample_task = tasks_test.sample_task()
        sample_size = 200
        x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        if torch.cuda.is_available():
            x_all = x_all.cuda()
            y_all = y_all.cuda()

        indices = np.arange(sample_size)
        np.random.shuffle(indices)
        support_indices = np.sort(indices[0:n_shot_test])

        query_indices = np.sort(indices[n_shot_test:])
        x_support = x_all[support_indices]
        y_support = y_all[support_indices]
        x_query = x_all[query_indices]
        y_query = y_all[query_indices]

        # Feed the support set
        z_support = net(x_support).detach()
        gp.train()
        fantasy_gp = gp.get_fantasy_model(inputs=z_support, targets=y_support)
        fantasy_gp.eval()

        # Evaluation on query set
        z_query = net(x_query).detach()
        mean = likelihood(fantasy_gp(z_query)).mean

        mse = criterion(mean, y_query)
        mse_list.append(mse.item())

    print("-------------------")
    print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")

    #Plotting
    for i in range(10):
        x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        if torch.cuda.is_available():
            x_all = x_all.cuda()
            y_all = y_all.cuda()

        query_indices = np.sort(indices[n_shot_test:])
        x_support = x_all[support_indices]
        y_support = y_all[support_indices]
        x_query = x_all[query_indices]
        y_query = y_all[query_indices]

        z_support = net(x_support).detach()
        gp.train()
        fantasy_gp = gp.get_fantasy_model(inputs=z_support, targets=y_support)
        fantasy_gp.eval()

        # Evaluation on all data
        z_all = net(x_all).detach()
        mean = likelihood(fantasy_gp(z_all)).mean
        lower, upper = likelihood(
            fantasy_gp(z_all)
        ).confidence_region()  # 2 standard deviations above and below the mean

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
        # ax.scatter(x_query, y_query, color='blue')
        
        # query points (predicted)
        ax.plot(np.squeeze(x_all), mean.detach().numpy(), color="red", linewidth=2.0)
        ax.fill_between(
            np.squeeze(x_all),
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
        plt.savefig("plot_DKT_due" + str(i) + ".png", dpi=300)


if __name__ == "__main__":
    #TODO: Make this configurable
    main()
