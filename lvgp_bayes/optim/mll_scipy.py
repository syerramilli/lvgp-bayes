import torch
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from collections import OrderedDict
from functools import reduce
from typing import Dict, List, Tuple, Optional, Union
from copy import deepcopy

def marginal_log_likelihood(model, add_prior: bool, use_likelihood_wrapper: bool = True):
    """Compute marginal log likelihood for GP models.

    Args:
        model: GP model instance
        add_prior: Whether to add prior contributions
        use_likelihood_wrapper: If True, wraps output with model.likelihood() (for standard GPs).
                                If False, assumes likelihood is included in model output (for FITC/VFE).

    Returns:
        Marginal log likelihood (scalar tensor)
    """
    output = model(*model.train_inputs)

    if use_likelihood_wrapper:
        # Standard GP: likelihood is separate
        out = model.likelihood(output).log_prob(model.train_targets)
    else:
        # Sparse GP (FITC/VFE): likelihood included in model output
        out = output.log_prob(model.train_targets)

    if add_prior:
        # add priors
        for _, module, prior, closure, _ in model.named_priors():
            out.add_(prior.log_prob(closure(module)).sum())

    # loss terms
    for added_loss_term in model.added_loss_terms():
        out.add_(added_loss_term.loss().sum())

    return out

class MLLObjective:
    """Helper class that wraps MLE/MAP objective function for scipy.optimize.

    Provides methods to pack/unpack model parameters and compute the objective
    function with gradients for use with scipy's L-BFGS-B optimizer.

    Args:
        model (GPR, LVGPR, or SparseLVGPR): GP model instance whose likelihood
            or posterior is to be optimized.
        add_prior (bool, optional): Whether to include prior contributions in
            the objective. If True, performs MAP estimation; if False, performs
            MLE. Defaults to True.
        use_likelihood_wrapper (bool, optional): If True, wraps model output
            with likelihood (for standard GPs). If False, assumes likelihood
            is included in model output (for sparse/FITC/VFE models).
            Defaults to True.

    Attributes:
        model: The GP model being optimized.
        add_prior (bool): Whether priors are included in objective.
        use_likelihood_wrapper (bool): Whether to use likelihood wrapper.
        param_shapes (OrderedDict): Shapes of model parameters for packing/unpacking.
    """
    def __init__(self, model, add_prior=True, use_likelihood_wrapper=True):
        self.model = model
        self.add_prior = add_prior
        self.use_likelihood_wrapper = use_likelihood_wrapper

        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        self.param_shapes = OrderedDict()
        
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                if len(parameters[n].size()) > 0:
                    self.param_shapes[n] = parameters[n].size()
                else:
                    self.param_shapes[n] = torch.Size([1])
    
    def pack_parameters(self) -> np.ndarray:
        """Pack model parameters into a 1D numpy array for scipy optimizer.

        Concatenates all trainable model parameters into a single flattened
        vector suitable for scipy.optimize.minimize.

        Returns:
            np.ndarray: Flattened parameter vector containing all trainable
                model hyperparameters.
        """
        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        
        return np.concatenate([parameters[n].data.numpy().ravel() for n in parameters])
    
    def unpack_parameters(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        """Unpack 1D parameter array into a dictionary of named tensors.

        Converts the flattened parameter vector from scipy optimizer back into
        a dictionary of named PyTorch tensors that can be loaded into the model.

        Args:
            x (np.ndarray): Flattened parameter vector from optimizer.

        Returns:
            dict: Dictionary mapping parameter names to PyTorch tensors with
                appropriate shapes for loading into model state dict.
        """
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self) -> np.ndarray:
        """Pack parameter gradients into a 1D numpy array for scipy optimizer.

        Concatenates gradients from all trainable parameters into a single
        flattened vector, matching the order of :meth:`pack_parameters`.

        Returns:
            np.ndarray: Flattened gradient vector as float64 array.
        """
        grads = []
        for _, p in self.model.named_parameters():
            if p.requires_grad:
                grad = p.grad.data.numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)

    def fun(self, x: np.ndarray, return_grad: bool = True) -> Union[float, Tuple[float, np.ndarray]]:
        """Objective function for scipy.optimize.minimize with optional gradients.

        Evaluates the negative log marginal likelihood (or posterior) at given
        parameters. Optionally computes gradients via automatic differentiation.

        Args:
            x (np.ndarray): Parameter vector (flattened hyperparameters).
            return_grad (bool, optional): If True, returns both objective and
                gradient. If False, returns only objective value. Defaults to True.

        Returns:
            float or tuple: If ``return_grad=False``, returns scalar objective value.
                If ``return_grad=True``, returns tuple of (objective, gradient) where
                gradient is a 1D numpy array computed via automatic differentiation.

        Note:
            The objective is the negative log marginal likelihood (MLE) or negative
            log posterior (MAP), suitable for minimization. Gradient computation
            uses PyTorch's autograd for exact derivatives.
        """
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        old_dict = self.model.state_dict()
        old_dict.update(state_dict)
        self.model.load_state_dict(old_dict)
        
        # zero the gradient
        self.model.zero_grad()
        obj = -marginal_log_likelihood(
            self.model, self.add_prior, self.use_likelihood_wrapper
        )  # negative sign to minimize
        
        if return_grad:
            # backprop the objective
            obj.backward()
            
            return obj.item(),self.pack_grads()
        
        return obj.item()

def fit_model_scipy(
    model,
    add_prior: bool = True,
    num_restarts: int = 5,
    jac: bool = True,
    theta0_list: Optional[List] = None,
    options: Dict = {},
    max_iter: int = 1000
) -> Tuple[List[OptimizeResult], float]:
    """Optimize GP model hyperparameters using scipy's L-BFGS-B optimizer.

    Performs Maximum A Posteriori (MAP) or Maximum Likelihood Estimation (MLE)
    for Gaussian Process models using multi-start optimization. Automatically
    detects model type and configures appropriate likelihood computation.

    Supports:
        - Standard GP models: GPR, LVGPR
        - Sparse GP models: SparseLVGPR with FITC/VFE approximations

    Args:
        model (GPR, LVGPR, or SparseLVGPR): GP model instance to optimize.
            Must implement ``reset_parameters()`` method if ``num_restarts > 0``.
        add_prior (bool, optional): If True, performs MAP estimation by including
            prior contributions. If False, performs MLE. Defaults to True.
        num_restarts (int, optional): Number of random restarts for multi-start
            optimization. More restarts improve chances of finding global optimum
            but increase runtime. Defaults to 5.
        jac (bool, optional): If True, uses automatic differentiation for exact
            gradients. If False, uses scipy's numerical differentiation (slower,
            less accurate). Defaults to True.
        theta0_list (list of np.ndarray, optional): List of initial parameter
            vectors for each restart. If provided, ``num_restarts`` is set to
            ``len(theta0_list) - 1``. If None, random initialization is used.
            Defaults to None.
        options (dict, optional): Dictionary of L-BFGS-B options passed to
            ``scipy.optimize.minimize``. If empty, uses default with
            ``{'maxfun': max_iter}``. Defaults to {}.
        max_iter (int, optional): Maximum number of function evaluations for
            L-BFGS-B. Only used if ``options`` does not specify 'maxfun'.
            Defaults to 1000.

    Returns:
        tuple: Two-element tuple containing:
            - **results** (list): List of ``scipy.optimize.OptimizeResult`` objects,
              one per restart. May also contain ``Exception`` objects if a restart failed.
            - **best_loss** (float): Best objective value found (negative log-likelihood
              or negative log-posterior).

    Example:
        Standard GP (LVGPR)::

            model = LVGPR(train_x, train_y, ...)
            results, best_loss = fit_model_scipy(model, num_restarts=5)

        Sparse GP (FITC/VFE)::

            model = SparseLVGPR(train_x, train_y, approx='FITC', ...)
            results, best_loss = fit_model_scipy(model, num_restarts=5)
            # Model type automatically detected!

        MLE instead of MAP::

            results, best_loss = fit_model_scipy(model, add_prior=False)

    Note:
        - Model type (standard vs sparse GP) is automatically detected via
          ``isinstance(model, SparseLVGPR)``. This determines whether to use
          likelihood wrapper in the objective computation.
        - The model is modified in-place with the best parameters found.
        - For reproducibility with random restarts, set random seeds before calling.
        - Each restart may produce different results due to non-convex optimization.
    """
    # Auto-detect model type
    from ..models.sparselvgp import SparseLVGPR

    is_sparse = isinstance(model, SparseLVGPR)
    use_likelihood_wrapper = not is_sparse

    # Set default options if not provided
    if not options:
        options = {'maxfun': max_iter}

    likobj = MLLObjective(model, add_prior, use_likelihood_wrapper)
    current_state_dict = deepcopy(likobj.model.state_dict())

    f_inc = np.inf
    # Output - Contains either optimize result objects or exceptions
    out = []

    
    if theta0_list is not None:
        num_restarts = len(theta0_list)-1
        old_dict = deepcopy(model.state_dict())
        old_dict.update(likobj.unpack_parameters(theta0_list[0]))
        model.load_state_dict(old_dict)

    for i in range(num_restarts+1):
        try:
            res = minimize(
                fun = likobj.fun,
                x0 = likobj.pack_parameters(),
                args=(True) if jac else (False),
                method = 'L-BFGS-B',
                jac=jac,
                bounds=None,
                options=options
            )
            out.append(res)
            
            if res.fun < f_inc:
                optimal_state = likobj.unpack_parameters(res.x)
                current_state_dict = deepcopy(likobj.model.state_dict())
                current_state_dict.update(optimal_state)
                f_inc = res.fun

        except Exception as e:
            out.append(e)
        
        likobj.model.load_state_dict(current_state_dict)
        if i < num_restarts:
            # reset parameters
            if theta0_list is None:
                model.reset_parameters()
            else:
                old_dict.update(likobj.unpack_parameters(theta0_list[i+1]))
                model.load_state_dict(old_dict)

    return out, f_inc