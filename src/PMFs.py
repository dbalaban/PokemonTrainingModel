import numpy as np
from data_structures import *

class IV_PMF:
    def __init__(self, prior: Optional[np.array] = None):
        if not prior is None:
            assert prior.shape == (6, 32)
            self.prior = prior
        else:
            self.prior = np.ones((6, 32)) / 32  # uniform prior

class EV_PMF:
    def __init__(self, priorT: Optional[np.array] = None, priorW: Optional[np.array] = None):
        self.max_ev = 252
        self.n_stats = 6
        # Max total with these rules: 2*252 + 6 = 510
        self.max_total_ev = 2 * self.max_ev + self.n_stats
        self.w_bins = 506

        # self.T - size 511 pmf over total EV values
        # self.W - 5x506 pmf over T-boundary, assumed conditional independence
        if not priorT is None:
            assert priorT.shape == ((self.max_total_ev+1),)
            self.T = priorT
        else:
            self.T = np.ones(((self.max_total_ev+1),)) / (self.max_total_ev+1)  # uniform prior
        
        if not priorW is None:
            assert priorW.shape == (5, 506)
            self.W = priorW
        else:
            self.W = np.ones((5, 506)) / 506  # uniform prior

    def get_corners(self, T):
        """
        Return 6 canonical vertices on the feasible EV hyperplane:
            sum(E_i) = T, 0 <= E_i <= max_ev
        specialized for n_stats = 6.

        The 6 vertices:
        - are valid allocations
        - depend on T
        - can be used with barycentric weights (sum=1, >=0)
            to define a simplex-based parameterization for some
            subset of the feasible slice at that T.

        Shape: (6, 6)  [6 vertices, each 6-dimensional].
        """
        assert T >= 0
        assert T <= self.max_total_ev

        # Degenerate: only all-zero is possible.
        if T == 0:
            return np.zeros((self.n_stats, self.n_stats), dtype=int)

        # Case 1: T <= max_ev
        # Vertices: put all T into exactly one stat.
        if T <= self.max_ev:
            V = np.zeros((self.n_stats, self.n_stats), dtype=int)
            idx = np.arange( self.n_stats)
            V[idx, idx] = T
            return V

        # Case 2: max_ev < T <= 2 * max_ev
        # Vertices: [max_ev, T-max_ev, 0, 0, 0, 0] and cyclic permutations.
        if T <= 2 *  self.max_ev:
            rest = T -  self.max_ev  # in (0, max_ev]
            base = np.zeros( self.n_stats, dtype=int)
            base[0] =  self.max_ev
            base[1] = rest
            # Roll base 0..5 to generate 6 symmetric vertices
            V = np.vstack([np.roll(base, shift=i) for i in range( self.n_stats)])
            return V

        # Case 3: 2 * max_ev < T <= max_total (510)
        # Vertices: [max_ev, max_ev, T-2*max_ev, 0, 0, 0] and cyclic perms.
        leftover = T - 2 *  self.max_ev  # in [1, 6]
        base = np.zeros( self.n_stats, dtype=int)
        base[0] =  self.max_ev
        base[1] =  self.max_ev
        base[2] = leftover
        V = np.vstack([np.roll(base, shift=i) for i in range( self.n_stats)])
        return V
    