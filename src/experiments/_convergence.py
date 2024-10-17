import torch
import numpy as np
import torch.distributions as TD
from ..utils import id_pretrain_model, train_diffusion_model
from ..utils import KL_train_distrib, KL_targ_distrib, energy_based_distance
from ..em import torchBatchItoEulerDistrib
from .manual_random import get_random_manager
from .exp_file_manager import Convergence_EFM, ConvergenceComparison_EFM
from ..icnn import DenseICNN
from ..diffusion import TargetedDiffusion
import time
import pickle
import math
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from matplotlib import pyplot as plt


def projS(X):

    e = torch.ones_like(X[0, :])
    # Compute the norm of each row
    row_norms = torch.sqrt(torch.sum(X**2, axis=1))

    # Avoid division by zero: if the norm is zero, replace it with one to avoid NaNs
    idx = torch.where(torch.abs(row_norms) < 1e-10)
    X_clone = X.clone()
    X_clone[idx] = e

    # Normalize each row
    return X_clone / torch.sqrt(torch.sum(X_clone**2, axis=1))[:, None]


distSq = lambda X: torch.sum((X - projS(X)) ** 2, axis=1)


def random_centers_distrib_generator(
    n, n_rand, centers_sample_distrib, std, device="cpu", dtype=torch.float32
):
    centers = centers_sample_distrib.sample((n_rand, n)).to(device, dtype)
    comp = TD.Independent(
        TD.Normal(
            centers,
            torch.tensor(
                [
                    std,
                ]
            ).to(device, dtype),
        ),
        1,
    )
    mix = TD.Categorical(
        torch.ones(
            n_rand,
        ).to(device, dtype)
    )
    target = TD.MixtureSameFamily(mix, comp)
    return target


def special_distrib_generator(
    dim, n_centers, dist, std, standardized=True, device="cpu"
):
    centers = np.zeros((n_centers, dim), dtype=np.float32)
    for d in range(dim):
        idx = np.random.choice(list(range(n_centers)), n_centers, replace=False)
        centers[:, d] += dist * idx
    centers -= dist * (n_centers - 1) / 2

    maps = np.random.normal(size=(n_centers, dim, dim)).astype(np.float32)
    maps /= np.sqrt((maps**2).sum(axis=2, keepdims=True))

    if standardized:
        mult = np.sqrt((centers**2).sum(axis=1).mean() + dim * std**2) / np.sqrt(dim)
        centers /= mult
        maps /= mult
    covars = np.matmul(maps, maps.transpose((0, 2, 1))) * (std**2)
    trc_centers = torch.tensor(centers, device=device, dtype=torch.float32)
    trc_covars = torch.tensor(covars, device=device, dtype=torch.float32)
    mv_normals = TD.MultivariateNormal(trc_centers, trc_covars)
    mix = TD.Categorical(
        torch.ones(
            n_centers,
        ).to(device)
    )
    target = TD.MixtureSameFamily(mix, mv_normals)
    return target


def create_stationary_distrib(config, device, dtype=torch.float32):
    if config["stationary_type"] == "cube":
        stationary_distrib = random_centers_distrib_generator(
            config["dim"],
            config["n_centers"],
            TD.Uniform(-config["target_span"], config["target_span"]),
            config["target_std"],
            device,
            dtype,
        )
    elif config["stationary_type"] == "normalized":
        if dtype != torch.float32:
            raise Exception('only float32 supported for "normalized" stationary')
        stationary_distrib = special_distrib_generator(
            config["dim"],
            config["n_centers"],
            config["target_span"],
            config["target_std"],
            device=device,
        )
    return stationary_distrib


class vMF2D_prior_to_set:
    def __init__(self, mu, kappa, rho, device):
        self.mu = torch.tensor(mu).to(device)
        self.kappa = torch.tensor(kappa).to(device)
        self.rho = torch.tensor(rho).to(device)
        self.device = device

        # compute normalization const
        def unnormalized_pdf(x, y):
            X = np.array([x, y])[np.newaxis, :]
            X = torch.tensor(X).to(device)
            updf = (
                torch.exp(
                    -(self.kappa / 2) * torch.sum((X - self.mu) ** 2, axis=1)
                    - (self.rho / 2) * distSq(X)
                )
                .cpu()
                .numpy()
            )
            return np.squeeze(updf)

        x_min, x_max = -np.inf, np.inf
        y_min, y_max = -np.inf, np.inf

        normalization_const, error = dblquad(
            unnormalized_pdf, x_min, x_max, lambda x: y_min, lambda x: y_max
        )

        self.normalization_const = torch.tensor(normalization_const).to(device)

    def log_prob(self, X):
        normalization_constant = self.normalization_const
        X = X.to(self.device)
        log_pdf = (
            -(self.kappa / 2) * torch.sum((X - self.mu) ** 2, axis=1)
            - (self.rho / 2) * distSq(X)
            - torch.log(torch.tensor(normalization_constant))
        )

        return log_pdf


def conv_comp_icnn_jko_dc_mix_gauss_targ_experiment(config):
    dim = config["dim"]
    device = config["device"]
    verbose = config["verbose"]
    discretization = config["discretization"]
    if discretization not in ["fb", "semi_fb"]:
        raise ValueError("unknown discretization.")
    t_train = config["dt"] * float(np.sum([n_iters for n_iters, _ in config["lr"]]))
    assert math.isclose(t_train, config["t_fin"], abs_tol=1e-5)

    for exp_number in config["exp_numbers"]:  # number of experiments
        if verbose:
            print(f"[ICNN jko dc convergence № {exp_number}] starts: dim: {dim}")
        # seed all random sources
        r_m = get_random_manager(config["random_key"], dim, exp_number)
        r_m.seed()
        # creates stationary distribution
        stationary_distrib = create_stationary_distrib(config, device)

        if discretization == "fb":
            G_component = lambda X: torch.unsqueeze(torch.sum(0 * X, dim=1), dim=1)
            H_component = lambda X: torch.unsqueeze(
                stationary_distrib.log_prob(X), dim=1
            )

        else:
            G_component = lambda X: torch.unsqueeze(
                torch.sum(X**2, dim=1) / config["target_std"] ** 2, dim=1
            )
            H_component = lambda X: G_component(X) + torch.unsqueeze(
                stationary_distrib.log_prob(X), dim=1
            )

        # create init sampler
        init_distrib = TD.MultivariateNormal(
            torch.zeros(dim).to(device),
            config["init_variance"] * torch.eye(dim).to(device),
        )

        # create base ICNN model
        batch_size = config["batch_size"]
        model_args = [
            dim,
            [
                config["layer_width"],
            ]
            * config["n_layers"],
        ]
        model_kwargs = {"rank": 5, "activation": "softplus", "batch_size": batch_size}
        D0 = DenseICNN(*model_args, **model_kwargs).to(device)

        # initialize the model
        for p in D0.parameters():
            p.data = torch.randn(p.shape, device=device, dtype=torch.float32) / np.sqrt(
                float(config["layer_width"])
            )

        # pretrain the model (to be identity function)
        D0 = id_pretrain_model(
            D0,
            init_distrib,
            lr=config["pretrain_lr"],
            n_max_iterations=4000,
            batch_size=batch_size,
            verbose=verbose,
        )

        diff = TargetedDiffusion(
            init_distrib,
            stationary_distrib,
            H_component,
            n_max_prop=config["n_max_prop"],
            dc_discretization=True,
            step_size=config["dt"],
            batch_size=batch_size,
        )
        X_test = init_distrib.sample((batch_size,)).view(-1, dim)

        t_tr_strt = time.perf_counter()
        kl_train = []
        # kl at the beginning
        curr_kl_train = KL_train_distrib(
            config["n_eval_samples"], diff, stationary_distrib
        )
        kl_train.append(curr_kl_train.item())
        for n_iters, lr in config["lr"]:
            diff, curr_kl_train = train_diffusion_model(
                diff,
                D0 if len(diff.Ds) == 0 else None,
                (model_args, model_kwargs),
                stationary_distrib,
                G_component=G_component,
                dc_discretization=True,
                n_steps=n_iters,
                step_iterations=config["n_step_iterations"],
                n_max_prop=config["n_max_prop"],
                step_size=config["dt"],
                batch_size=batch_size,
                X_test=X_test,
                lr=lr,
                device=device,
                plot_loss=False,
                verbose=verbose,
            )
            kl_train = kl_train + curr_kl_train
        t_tr_elaps = time.perf_counter() - t_tr_strt

        if verbose:
            print(f"[ICNN jko dc convergence № {exp_number}]: diffusion estimation")
            # true distribution at diff_tstp timestep
        exp_results = {}
        t_est_strt = time.perf_counter()
        # KL(model||target)
        curr_kl_train, diff_X = KL_train_distrib(
            config["n_eval_samples"], diff, stationary_distrib, ret_diff_sample=True
        )
        exp_results["kl_train_last"] = curr_kl_train.item()
        exp_results["kl_train_process"] = kl_train
        t_est_elaps = time.perf_counter() - t_est_strt
        exp_results["time_train"] = t_tr_elaps
        exp_results["time_est"] = t_est_elaps

        # draw figs
        xylim = 6
        nn = 200
        x = np.linspace(-xylim, xylim, nn)
        y = np.linspace(-xylim, xylim, nn)
        X, Y = np.meshgrid(x, y)
        model_samples = diff.sample(500).cpu()
        model_samples_x = model_samples[:, 0]
        model_samples_y = model_samples[:, 1]
        valid_indices = np.where(
            (model_samples_x >= -xylim)
            & (model_samples_x <= xylim)
            & (model_samples_y >= -xylim)
            & (model_samples_y <= xylim)
        )
        model_samples_x = model_samples_x[valid_indices]
        model_samples_y = model_samples_y[valid_indices]

        Z = torch.zeros(nn, nn)
        for i in range(nn):
            for j in range(nn):
                Z[i, j] = stationary_distrib.log_prob(
                    torch.tensor([X[i, j], Y[i, j]]).to(device=device)
                ).exp()
        plt.figure()
        plt.contourf(X, Y, Z, cmap="viridis")  # Change levels for more or less curves
        plt.scatter(model_samples_x, model_samples_y, color="red", alpha=0.3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(
            "figs/samples_" + discretization + str(exp_number) + ".png", dpi=300
        )

        exp_results["log_prob_true"] = Z
        exp_results["X"] = X
        exp_results["Y"] = Y
        exp_results["samples_x"] = model_samples_x
        exp_results["samples_y"] = model_samples_y

        with open(
            "pickles/exp_results_"
            + discretization
            + "_dim_"
            + str(dim)
            + "_nexp_"
            + str(exp_number)
            + ".pickle",
            "wb",
        ) as f:
            pickle.dump(exp_results, f)


def conv_comp_em_mix_gauss_targ_experiment(config):
    dim = config["dim"]
    exp_numbers = config["exp_numbers"]
    em_dt = config["dt"]
    n_particles = config["n_particles"]
    n_eval_samples = config["n_eval_samples"]
    if n_eval_samples == -1:
        n_eval_samples = n_particles
    verbose = config["verbose"]
    device = config["device"]

    for exp_number in exp_numbers:
        if verbose:
            print(f"[EM convergence № {exp_number}] starts: dim: {dim}")
        # seed all random sources
        r_m = get_random_manager(config["random_key"], dim, exp_number)
        r_m.seed()
        # creates stationary distribution
        stationary_distrib = create_stationary_distrib(config, device)

        # create init sampler
        init_distrib = TD.MultivariateNormal(
            torch.zeros(dim).to(device),
            config["init_variance"] * torch.eye(dim).to(device),
        )

        t_tr_strt = time.perf_counter()
        # sample particles from the initial distribution
        x0 = init_distrib.sample((n_particles,))

        final_x = (
            torchBatchItoEulerDistrib(stationary_distrib, x0, em_dt, config["t_fin"])
            .cpu()
            .numpy()
        )
        if verbose:
            print(f"[EM convergence № {exp_number}] particles have been simulated")
        final_distrib = gaussian_kde(final_x.transpose())
        if verbose:
            print(f"[EM convergence № {exp_number}] kde has been built")
        t_tr_elaps = time.perf_counter() - t_tr_strt

        exp_results = {}
        t_ev_strt = time.perf_counter()
        # KL wrt train est.
        em_sample = final_distrib.resample(n_eval_samples).T
        exp_results["kl_train_last"] = KL_train_distrib(
            em_sample, final_distrib, stationary_distrib, device=device
        )

        t_ev_elaps = time.perf_counter() - t_ev_strt

        exp_results["time_train"] = t_tr_elaps
        exp_results["time_est"] = t_ev_elaps

        # draw figs
        xylim = 6
        nn = 200
        x = np.linspace(-xylim, xylim, nn)
        y = np.linspace(-xylim, xylim, nn)
        X, Y = np.meshgrid(x, y)
        model_samples = final_distrib.resample(500).T
        model_samples_x = model_samples[:, 0]
        model_samples_y = model_samples[:, 1]
        valid_indices = np.where(
            (model_samples_x >= -xylim)
            & (model_samples_x <= xylim)
            & (model_samples_y >= -xylim)
            & (model_samples_y <= xylim)
        )
        model_samples_x = model_samples_x[valid_indices]
        model_samples_y = model_samples_y[valid_indices]

        Z = torch.zeros(nn, nn)
        for i in range(nn):
            for j in range(nn):
                Z[i, j] = stationary_distrib.log_prob(
                    torch.tensor([X[i, j], Y[i, j]]).to(device=device)
                ).exp()
        plt.figure()
        plt.contourf(X, Y, Z, cmap="viridis")  # Change levels for more or less curves
        plt.scatter(model_samples_x, model_samples_y, color="red", alpha=0.3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("figs/samples_EM_" + str(exp_number) + ".png", dpi=300)

        exp_results["log_prob_true"] = Z
        exp_results["X"] = X
        exp_results["Y"] = Y
        exp_results["samples_x"] = model_samples_x
        exp_results["samples_y"] = model_samples_y

        with open(
            "pickles/exp_results_EM"
            + "_dim_"
            + str(dim)
            + "_nexp_"
            + str(exp_number)
            + ".pickle",
            "wb",
        ) as f:
            pickle.dump(exp_results, f)


def distance_to_set_prior_experiment(config):
    dim = config["dim"]
    device = config["device"]
    verbose = config["verbose"]
    discretization = config["discretization"]

    # posterior params
    rho = config["rho"]
    kappa = config["kappa"]
    mu = torch.tensor([config["mu1"], config["mu2"]]).to(device)

    if discretization not in ["fb", "semi_fb"]:
        raise ValueError("unknown discretization.")
    t_train = config["dt"] * float(np.sum([n_iters for n_iters, _ in config["lr"]]))
    assert math.isclose(t_train, config["t_fin"], abs_tol=1e-5)

    for exp_number in config["exp_numbers"]:  # number of experiments
        if verbose:
            print(f"[ICNN jko dc convergence № {exp_number}] starts: dim: {dim}")
        # seed all random sources
        r_m = get_random_manager(config["random_key"], dim, exp_number)
        r_m.seed()
        # creates stationary distribution
        stationary_distrib = vMF2D_prior_to_set(
            mu=mu, kappa=kappa, rho=rho, device=device
        )

        if discretization == "fb":
            G_component = lambda X: torch.unsqueeze(torch.sum(0 * X, dim=1), dim=1)
            H_component = lambda X: -torch.unsqueeze(
                (kappa / 2) * torch.sum((X - mu) ** 2, axis=1) + (rho / 2) * distSq(X),
                dim=1,
            )
        else:
            G_component = lambda X: torch.unsqueeze(
                (kappa / 2) * torch.sum((X - mu) ** 2, axis=1)
                + (rho / 2) * torch.sum(X**2, axis=1),
                dim=1,
            )
            H_component = lambda X: torch.unsqueeze(
                (rho / 2) * torch.sum(X**2, axis=1) - (rho / 2) * distSq(X), dim=1
            )

        # create init sampler
        init_distrib = TD.MultivariateNormal(
            torch.zeros(dim).to(device),
            config["init_variance"] * torch.eye(dim).to(device),
        )

        # create base ICNN model
        batch_size = config["batch_size"]
        model_args = [
            dim,
            [
                config["layer_width"],
            ]
            * config["n_layers"],
        ]
        model_kwargs = {"rank": 5, "activation": "softplus", "batch_size": batch_size}
        D0 = DenseICNN(*model_args, **model_kwargs).to(device)

        # initialize the model
        for p in D0.parameters():
            p.data = torch.randn(p.shape, device=device, dtype=torch.float32) / np.sqrt(
                float(config["layer_width"])
            )

        # pretrain the model (to be identity function)
        D0 = id_pretrain_model(
            D0,
            init_distrib,
            lr=config["pretrain_lr"],
            n_max_iterations=4000,
            batch_size=batch_size,
            verbose=verbose,
        )

        diff = TargetedDiffusion(
            init_distrib,
            stationary_distrib,
            H_component,
            n_max_prop=config["n_max_prop"],
            dc_discretization=True,
            step_size=config["dt"],
            batch_size=batch_size,
        )
        X_test = init_distrib.sample((batch_size,)).view(-1, dim)

        t_tr_strt = time.perf_counter()
        kl_train = []
        # kl at the beginning
        curr_kl_train = KL_train_distrib(
            config["n_eval_samples"], diff, stationary_distrib
        )
        kl_train.append(curr_kl_train.item())
        for n_iters, lr in config["lr"]:
            diff, curr_kl_train = train_diffusion_model(
                diff,
                D0 if len(diff.Ds) == 0 else None,
                (model_args, model_kwargs),
                stationary_distrib,
                G_component=G_component,
                dc_discretization=True,
                n_steps=n_iters,
                step_iterations=config["n_step_iterations"],
                n_max_prop=config["n_max_prop"],
                step_size=config["dt"],
                batch_size=batch_size,
                X_test=X_test,
                lr=lr,
                device=device,
                plot_loss=False,
                verbose=verbose,
            )
            kl_train = kl_train + curr_kl_train
        t_tr_elaps = time.perf_counter() - t_tr_strt

        # plot kl_train
        plt.figure()
        plt.plot(kl_train)
        plt.ylabel("KL")
        plt.savefig("figs/KL_" + discretization + str(exp_number) + ".png")
        if verbose:
            print(f"[ICNN jko dc convergence № {exp_number}]: diffusion estimation")
            # true distribution at diff_tstp timestep
        exp_results = {}
        t_est_strt = time.perf_counter()
        # KL(model||target)
        curr_kl_train, diff_X = KL_train_distrib(
            config["n_eval_samples"], diff, stationary_distrib, ret_diff_sample=True
        )
        exp_results["kl_train_last"] = curr_kl_train.item()
        exp_results["kl_train_process"] = kl_train
        t_est_elaps = time.perf_counter() - t_est_strt
        exp_results["time_train"] = t_tr_elaps
        exp_results["time_est"] = t_est_elaps
        # file_manager.save(exp_results, exp_number)

        # draw figs
        xylim = 2
        nn = 200
        x = np.linspace(-xylim, xylim, nn)
        y = np.linspace(-xylim, xylim, nn)
        X, Y = np.meshgrid(x, y)
        model_samples = diff.sample(500).cpu()
        model_samples_x = model_samples[:, 0]
        model_samples_y = model_samples[:, 1]
        valid_indices = np.where(
            (model_samples_x >= -xylim)
            & (model_samples_x <= xylim)
            & (model_samples_y >= -xylim)
            & (model_samples_y <= xylim)
        )
        model_samples_x = model_samples_x[valid_indices]
        model_samples_y = model_samples_y[valid_indices]

        Z = torch.zeros(nn, nn)
        for i in range(nn):
            for j in range(nn):
                Z[i, j] = (
                    stationary_distrib.log_prob(
                        torch.tensor([X[i, j], Y[i, j]])[None, :]
                    )
                    .exp()
                    .cpu()
                )
        plt.figure()
        plt.contourf(
            X, Y, Z, cmap="viridis", levels=200
        )  # Change levels for more or less curves
        # plt.scatter(model_samples_x, model_samples_y, color="blue", alpha=0.3, s=1)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(
            "figs/vmf_truepdf_" + discretization + str(exp_number) + ".png", dpi=300
        )

        plt.figure()
        plt.hist2d(
            model_samples_x.numpy(),
            model_samples_y.numpy(),
            bins=50,
            cmap="Greys",
            range=[(-xylim, xylim), (-xylim, xylim)],
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(
            "figs/vmf_estpdf_" + discretization + str(exp_number) + ".png", dpi=300
        )
        # plt.cm.jet

        exp_results["log_prob_true"] = Z
        exp_results["X"] = X
        exp_results["Y"] = Y
        exp_results["samples_x"] = model_samples_x
        exp_results["samples_y"] = model_samples_y

        with open(
            "pickles/vmf_exp_results_"
            + discretization
            + "_dim_"
            + str(dim)
            + "_nexp_"
            + str(exp_number)
            + ".pickle",
            "wb",
        ) as f:
            pickle.dump(exp_results, f)
