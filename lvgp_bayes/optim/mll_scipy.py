import torch
import numpy as np
from gpytorch import settings as gptsettings
from scipy.optimize import minimize,OptimizeResult
from collections import OrderedDict
from functools import reduce
from typing import Dict,List,Tuple,Optional,Union
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
    """Helper class that wraps MLE/MAP objective function to be called by scipy.optimize.

    :param model: A GP model instance (e.g., GPR, LVGPR, SparseLVGPR) whose
        likelihood/posterior is to be optimized.
    :type model: models.GPR or models.SparseLVGPR

    :param add_prior: Whether to include prior contributions in the objective.
    :type add_prior: bool

    :param use_likelihood_wrapper: If True, wraps model output with likelihood
        (for standard GPs). If False, assumes likelihood is included in model
        output (for sparse/FITC/VFE models). Defaults to True.
    :type use_likelihood_wrapper: bool
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
        """Returns the current hyperparameters in vector form for the scipy optimizer

        :return Current hyperparameters in a 1-D array representation
        :rtype: np.ndarray
        """
        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        
        return np.concatenate([parameters[n].data.numpy().ravel() for n in parameters])
    
    def unpack_parameters(self, x:np.ndarray) -> torch.Tensor:
        """Convert hyperparameters specifed as a 1D array to a named parameter dictionary
        that can be imported by the model

        :param x: Hyperparameters in flattened vector form
        :type x: np.ndarray

        :returns: A dictionary of hyperparameters
        :rtype: Dict
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

    def pack_grads(self) -> None:
        """Concatenate gradients from the parameters to 1D numpy array
        """
        grads = []
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                grad = p.grad.data.numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)

    def fun(self, x:np.ndarray,return_grad=True) -> Union[float,Tuple[float,np.ndarray]]:
        """Function to be passed to `scipy.optimize.minimize`,

        :param x: Hyperparameters in 1D representation
        :type x: np.ndarray

        :param return_grad: Return gradients computed via automatic differentiation if 
            `True`. Defaults to `True`.
        :type return_grad: bool, optional

        Returns:
            One of the following, depending on `return_grad`
                - If `return_grad`=`False`, returns only the objective
                - If `return_grad`=`False`, returns a two-element tuple containing:
                    - the objective
                    - a numpy array of the gradients of the objective wrt the 
                      hyperparameters computed via automatic differentiation
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
    """Optimize the likelihood/posterior of a GP model using `scipy.optimize.minimize`.

    This function automatically detects the model type and uses the appropriate
    likelihood computation method. Works with both standard GP models (GPR, LVGPR)
    and sparse GP models (SparseLVGPR with FITC/VFE approximations).

    :param model: A model instance (e.g., GPR, LVGPR, SparseLVGPR). The model must
        implement a `.reset_parameters()` method if `num_restarts > 0`.
    :type model: models.GPR or models.SparseLVGPR

    :param add_prior: Whether to include prior contributions in the objective.
        Defaults to True (MAP estimation). Set to False for MLE.
    :type add_prior: bool, optional

    :param num_restarts: The number of times to restart the local optimization from a
        new starting point. Defaults to 5.
    :type num_restarts: int, optional

    :param jac: Use automatic differentiation to compute gradients if `True`. If `False`, uses
        scipy's numerical differentiation mechanism. Defaults to `True`.
    :type jac: bool, optional

    :param theta0_list: Optional list of initial parameter vectors. If provided,
        `num_restarts` is set to `len(theta0_list) - 1`.
    :type theta0_list: Optional[List], optional

    :param options: A dictionary of `L-BFGS-B` options to be passed to `scipy.optimize.minimize`.
        If not provided, uses default with `maxfun: max_iter`.
    :type options: dict, optional

    :param max_iter: Maximum number of iterations for L-BFGS-B. Only used if `options`
        does not specify 'maxfun'. Defaults to 1000.
    :type max_iter: int, optional

    Returns:
        A two-element tuple with the following elements:
            - a list of optimization result objects, one for each starting point.
            - the best (negative) log-likelihood/log-posterior found

    :rtype: Tuple[List[OptimizeResult], float]

    Example:
        Standard GP (LVGPR)::

            model = LVGPR(train_x, train_y, ...)
            results, best_loss = fit_model_scipy(model, num_restarts=5)

        Sparse GP (FITC/VFE)::

            model = SparseLVGPR(train_x, train_y, approx='FITC', ...)
            results, best_loss = fit_model_scipy(model, num_restarts=5)
            # Model type automatically detected!

    Note:
        The function automatically detects whether the model is a sparse GP
        (SparseLVGPR) or standard GP (GPR/LVGPR) and adjusts the likelihood
        computation accordingly. No manual configuration needed!
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