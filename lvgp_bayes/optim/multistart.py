import numpy as np
from scipy.optimize import minimize,Bounds
from joblib import Parallel,delayed

from ..utils.input_space import _latinhypercube_sample

class MultiStartOptimizer:
    '''
    Base class for multi-start numerical optimization using L-BFGS-B 

    Parameters
    -----------------
    obj: callable
        The objective function to be minimized
    
    lb,ub: array-like,
        Lower and upper bounds on independent variables. Each array must have the same 
        size as x. All variables need to have finite bounds to be specified.
    
    num_starts: int,
        The number of starting points from which the L-BFGS-B search is initialized
    
    x0: array-like, optional
        A specific starting point. Useful when iteratively updating a model. Note that
        `num_starts` excludes x0.
    
    jac: {callable,bool}, optional
        Either a method for computing the gradient vector or a boolean variable
        indicating whether `obj` returns the gradient vector along with the objective.
        Specify `None` (default) if using numerical gradients (computed by scipy).
    
    num_jobs: int, optional
        The number of jobs to run in parallel
    
    rng: np.random.RandomState, optional,
        Random number generator
    
    Attributes
    -----------------
    x_inc: np.ndarray
        The best solution found across all the starts
    
    f_inc: float
        Value of the objective function at the best solution
    '''
    def __init__(
        self,
        obj,
        lb,ub,
        num_starts,
        x0=None,
        jac=None,
        num_jobs=1,
        rng=None,
        **kwargs):
        
        self.obj=obj # objective
        self.lb = lb # bounds on the variables
        self.ub = ub
        self.x0 = x0 # initial guess (optional)
        self.jac = jac # specifiy the jacobian

        self.num_starts = num_starts

        # Argument for parallelizing
        self.num_jobs = num_jobs
        
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
        
        self.local_options = kwargs.pop('options',{})

    def run(self):
        '''
        Peform the multi-start optimization procedure

        Returns
        ----------
        x_inc: np.ndarray
            The best solution found across all the starts
        '''
        self.num_evals = 0
        
        # generate initial starting points
        x_init = self.generate_init()
        
        self.multi_local = Parallel(n_jobs=self.num_jobs,verbose=0)(
            delayed(self.local_search)(x) for x in x_init
        )
        
        self.x_inc = None;self.f_inc = np.inf
        for x,f_x,num_local_evals in self.multi_local:
            if not isinstance(f_x,float):
                continue
            if f_x < self.f_inc:
                self.x_inc = x.copy()
                self.f_inc = f_x
        
        # scale the variables back before returning
        return self.x_inc

    def generate_init(self):
        '''
        Generate `num_starts` points using Latin hypercube sampling

        Returns
        -----------
        x_init: np.ndarray
            Matrix of starting points with each row representing one sample point
        '''
        x_init = _latinhypercube_sample(self.rng,ndim=self.lb.shape[0],size=self.num_starts)
        x_init = self.lb + (self.ub-self.lb)*x_init
        if self.x0 is not None:
            x_init = np.row_stack([
                self.x0.T,x_init
            ])
        return x_init

    def local_search(self,x0):
        '''
        Perform L-BFGS-B search from a given starting point

        Parameters
        -------------
        x0: array-like
            The starting point. The dimensions should match those of `lb` and `ub`.
        
        Returns
        --------------
        x: np.ndarray
            The local minumum
        
        f: float
            Value of the objective at the local minimum
        
        num_tries: int
            Number of evaluations of the objectibe function
        '''
        try:
            res = minimize(
                fun = self.obj,
                x0 = x0,
                method='L-BFGS-B',
                jac=self.jac,
                bounds=Bounds(self.lb,self.ub),
                options=self.local_options
            )
            x = res.x
            f = res.fun
            num_tries = res.nfev
        except:
            x = x0.copy()
            try:
                f = self.obj(x0)
            except:
                f = np.inf
            num_tries = 2 # can be larger but not maintaining count 
        
        return x,f,num_tries