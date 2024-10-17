import torch
import torch.nn as nn
from multipledispatch import dispatch
import numpy as np
from collections.abc import Iterable
import torch.autograd as autograd


class Diffusion:

    @staticmethod
    def negative_entropy_gain(push_X, X):
        """The negative gain of the entropy between pushforward and original distribution"""
        hessian = torch.cat(
            [
                torch.autograd.grad(
                    outputs=push_X[:, d],
                    inputs=X,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    grad_outputs=torch.ones(X.size()[0]).float().to(X),
                )[0][:, None, :]
                for d in range(X.size(1))
            ],
            dim=1,
        )
        #         signs, logabsdet = torch.slogdet(hessian)
        return -torch.logdet(hessian)

    def __init__(
        self,
        init_distrib,
        H_component,
        n_max_prop=None,
        dc_discretization=False,
        step_size=None,
        batch_size=None,
    ):
        self.init_distrib = init_distrib
        self.init_is_trch_distrib = isinstance(
            self.init_distrib, torch.distributions.Distribution
        )
        self.H_component = H_component
        self.n_max_prop = n_max_prop
        self.dc_discretization = dc_discretization
        self.step_size = step_size
        self.batch_size = batch_size
        self.Ds = []

    def negative_entropy_gain_no_grad(self, X, n_max_prop, return_push_X=False, D=None):

        if n_max_prop is None:
            push_X = X + self.step_size * self.H_push(X) if D is None else D.push(X)
            neg = Diffusion.negative_entropy_gain(push_X, X).detach()
            if return_push_X:
                return neg, push_X.detach()
            return neg

        if return_push_X:
            push_X = torch.zeros_like(X)
        neg = torch.zeros(X.size(0), dtype=X.dtype, device=X.device)
        for i_prop in range(0, X.size(0), n_max_prop):
            X_prop = X[i_prop : i_prop + n_max_prop]
            X_prop.requires_grad_(True)
            push_X_prop = (
                X_prop + self.step_size * self.H_push(X_prop)
                if D is None
                else D.push(X_prop)
            )
            if return_push_X:
                push_X[i_prop : i_prop + n_max_prop] = push_X_prop
            neg[i_prop : i_prop + n_max_prop] = Diffusion.negative_entropy_gain(
                push_X_prop, X_prop
            ).detach()
        if return_push_X:
            return neg, push_X.detach()
        return neg

    def H_push(self, input, create_graph=True, retain_graph=True):
        """
        Pushes input through grad H. By default preserves the computational graph.
        # Apply to small batches.
        """
        if len(input) <= self.batch_size:  # if X is small, push X
            output = autograd.grad(
                outputs=self.H_component(input),
                inputs=input,
                create_graph=create_graph,
                retain_graph=retain_graph,
                only_inputs=True,
                grad_outputs=torch.ones_like(input[:, :1], requires_grad=False),
            )[0]
            return output
        else:  # if X is large, iteratively push its small batches
            output = torch.zeros_like(input, requires_grad=False)
            for j in range(0, input.size(0), self.batch_size):
                output[j : j + self.batch_size] = self.H_push(
                    input[j : j + self.batch_size],
                    create_graph=create_graph,
                    retain_graph=retain_graph,
                )
            return output

    def sample_init(self, n):
        X = self.init_distrib.sample((n,) if self.init_is_trch_distrib else n)
        if len(X.shape) == 1:
            return X.view(-1, 1)
        return X

    def sample(self, n, save_interm=False, nu=False):
        X = self.sample_init(n)
        return self.propagate(
            X, save_interm=save_interm, nu=nu
        )  # sample from the last mu_n

    def propagate(
        self, X, save_interm=False, nu=True
    ):  # if nu, propagate to nu_{n+1}, otherwise propagate to mu_n
        if self.dc_discretization:
            if not save_interm:
                for j in range(len(self.Ds)):
                    X.requires_grad_(True)
                    X = X + self.step_size * self.H_push(X)
                    X.requires_grad_(True)
                    X = self.Ds[j].push_nograd(X)
                if nu:
                    X.requires_grad_(True)
                    X = X + self.step_size * self.H_push(X)
                return X
            else:
                Xs = []
                Ys = []
                Xs.append(X.detach().clone())
                for j in range(len(self.Ds)):
                    X.requires_grad_(True)
                    X = X + self.step_size * self.H_push(X)
                    Ys.append(X.detach().clone())

                    X.requires_grad_(True)
                    X = self.Ds[j].push_nograd(X)
                    Xs.append(X.detach().clone())
                if nu:
                    X.requires_grad_(True)
                    X = X + self.step_size * self.H_push(X)
                    Ys.append(X.detach().clone())

                return Xs, Ys
        else:
            if not save_interm:
                for j in range(len(self.Ds)):
                    X.requires_grad_(True)
                    X = self.Ds[j].push_nograd(X)
                return X
            else:
                Xs = []
                Xs.append(X.detach().clone())
                for j in range(len(self.Ds)):
                    X.requires_grad_(True)
                    X = self.Ds[j].push_nograd(X)
                    Xs.append(X.detach().clone())
                return Xs

    @dispatch(torch.Tensor)
    def mc_entropy(
        self,
        X,
        ignore_init=False,
        return_X_transformed=False,
        try_true_init_entropy=False,
    ):
        entropy = 0.0
        if not ignore_init:
            if try_true_init_entropy:
                try:
                    entropy -= self.init_distrib.entropy()
                except:
                    try:
                        entropy += self.init_distrib.log_prob(X).mean()
                    except:
                        raise Exception("no way to estimate 'init_distrib' entropy")
            else:
                try:
                    entropy += self.init_distrib.log_prob(X).mean()
                except:
                    raise Exception("no way to estimate 'init_distrib' entropy")
        for i in range(len(self.Ds)):  # still incorrect for DC
            if self.dc_discretization:
                X.requires_grad_(True)
                # push through (I + gamma * grad H)
                neg, push_X = self.negative_entropy_gain_no_grad(
                    X, self.n_max_prop, True, D=None
                )
                entropy += neg.mean()
                X = push_X

            X.requires_grad_(True)
            # push through gradient of neural network
            neg, push_X = self.negative_entropy_gain_no_grad(
                X, self.n_max_prop, True, D=self.Ds[i]
            )
            entropy += neg.mean()
            X = push_X
        if not return_X_transformed:
            return entropy
        return entropy, X

    @dispatch(int)
    def mc_entropy(self, n, ignore_init=False, return_X_transformed=False):
        X = self.sample_init(n)
        return self.mc_entropy(
            X, ignore_init=ignore_init, return_X_transformed=return_X_transformed
        )

    def log_prob_trace(self, X_trace, ignore_init=False, backward_order=True):
        if not backward_order:
            X_trace = list(reversed(X_trace))
        _log_prob = 0.0
        if not ignore_init:
            _log_prob = self.init_distrib.log_prob(X_trace[-1]).view(-1)
        for i in range(len(self.Ds)):
            n_xs = -i - 1
            X_trace[n_xs].requires_grad_(True)
            neg = self.negative_entropy_gain_no_grad(
                X_trace[n_xs], self.n_max_prop, False, D=self.Ds[i]
            )
            _log_prob += neg.view(-1)
        return _log_prob

    # def log_prob(
    #     self,
    #     X,
    #     ignore_init=False,
    #     return_X_proto=False,
    #     method="grad_gd",
    #     double_precision=False,
    #     max_iter=1000,
    #     grad_tol=1e-5,
    #     no_progress_stop=5,
    #     min_lambda=1e-8,
    #     verbose=False,
    # ):

    #     assert method == "grad_gd", "method {} not yet implemented".format(method)

    #     if method == "grad_gd":

    #         def func(D, X_opt, X_ref):
    #             return D(X_opt) - (X_opt * X_ref).sum(-1).view(-1, 1)

    #         def func_grad(D, X_opt, X_ref):
    #             X_opt.requires_grad_(True)
    #             return D.push_nograd(X_opt) - X_ref

    #         def func_grad_norm(D, X_opt, X_ref):
    #             return torch.norm(func_grad(D, X_opt, X_ref).detach(), dim=-1)

    #         Ds = self.Ds
    #         if double_precision:
    #             Ds = [Ds[i].double() for i in range(len(Ds))]
    #             X = X.double()

    #         Xs_backward = []
    #         Xs_backward.append(X)

    #         for i in range(len(Ds)):
    #             if verbose:
    #                 print("Ds[{}] pushback starts".format(len(Ds) - i - 1))
    #             ds_num = -i - 1
    #             Xs_curr = Xs_backward[-1].clone()
    #             Xs_prev = Xs_curr.clone()
    #             Xs_prev.detach_()
    #             lr_base = 1.0
    #             j = 0
    #             max_grad_norm_history = []
    #             mask = torch.arange(0, X.size(0), dtype=int)
    #             while True:
    #                 _Xs_prev = Xs_prev[mask]  # we optimize
    #                 _Xs_prev.requires_grad_(True)
    #                 _Xs_curr = Xs_curr[mask]  # reference
    #                 _Xs_prev.requires_grad_(True)
    #                 _grad = func_grad(Ds[ds_num], _Xs_prev, _Xs_curr).detach()
    #                 prev_grad_norms = torch.norm(_grad, dim=-1)
    #                 _lambdas = (lr_base / torch.sqrt(prev_grad_norms + 1e-6)).view(
    #                     -1, 1
    #                 )
    #                 while True:
    #                     _new_Xs = _Xs_prev - _lambdas * _grad
    #                     curr_grad_norms = func_grad_norm(Ds[ds_num], _new_Xs, _Xs_curr)
    #                     diff = prev_grad_norms - curr_grad_norms
    #                     if torch.sum(diff <= 0.0) == 0:
    #                         break
    #                     if torch.min(_lambdas) < min_lambda:
    #                         break
    #                     _lambdas[diff <= 0.0] *= 0.5

    #                 _Xs_prev = _Xs_prev - _lambdas * _grad
    #                 final_grad_norms = func_grad_norm(Ds[ds_num], _Xs_prev, _Xs_curr)
    #                 max_grad_norm_history.append(final_grad_norms.max())
    #                 acheve_mask = final_grad_norms < grad_tol
    #                 Xs_prev[mask] = _Xs_prev.detach()
    #                 mask = mask[~acheve_mask]
    #                 # print(len(mask))
    #                 if len(mask) == 0:
    #                     if verbose:
    #                         print("pushback {} has taken {} iters".format(i, j))
    #                         print(
    #                             "max grad diff: ",
    #                             func_grad_norm(Ds[ds_num], Xs_prev, Xs_curr).max(),
    #                         )
    #                     break
    #                 if j > max_iter:
    #                     if verbose:
    #                         print("stopped since max_iter acheved")
    #                         print("N not converged: ", len(mask))
    #                         print(
    #                             "max grad diff: ",
    #                             func_grad_norm(Ds[ds_num], Xs_prev, Xs_curr).max(),
    #                         )
    #                     break
    #                 if j > no_progress_stop:
    #                     if (
    #                         np.max(max_grad_norm_history[-no_progress_stop:])
    #                         - np.min(max_grad_norm_history[-no_progress_stop:])
    #                         < 1e-16
    #                     ):
    #                         if verbose:
    #                             print("stopped since no progress acheved")
    #                             print("N not acheved: ", len(mask))
    #                             print("pushback {} has taken {} iters".format(i, j))
    #                             print(
    #                                 "max grad diff: ",
    #                                 func_grad_norm(Ds[ds_num], Xs_prev, Xs_curr).max(),
    #                             )
    #                         break
    #                 j += 1
    #             Xs_backward.append(Xs_prev.detach())

    #         _log_prob = self.log_prob_trace(Xs_backward, ignore_init=ignore_init)

    #         if double_precision:
    #             Ds = [Ds[i].float() for i in range(len(Ds))]
    #         if not return_X_proto:
    #             return _log_prob.detach()
    #         return Xs_backward[-1], _log_prob.detach()


class TargetedDiffusion(Diffusion):

    def __init__(
        self,
        init_distrib,
        target_distrib,
        H_component=None,
        n_max_prop=None,
        dc_discretization=False,
        step_size=None,
        batch_size=None,
    ):
        if dc_discretization:
            if step_size is None or batch_size is None:
                ValueError("Have to specify step_size and batch_size.")
        super().__init__(
            init_distrib,
            H_component,
            n_max_prop=n_max_prop,
            dc_discretization=dc_discretization,
            step_size=step_size,
            batch_size=batch_size,
        )
        self.target_distrib = target_distrib
