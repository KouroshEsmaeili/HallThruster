from .flux_functions import FluxFunction, rusanov
from .limiters import SlopeLimiter, van_leer

class HyperbolicScheme:

    def __init__(self, flux_function:FluxFunction=None, limiter:SlopeLimiter=None, reconstruct=True):
        if flux_function is None:
            flux_function = rusanov  # or rusanov

        if limiter is None:
            limiter = van_leer

        self.flux_function = flux_function
        self.limiter = limiter
        self.reconstruct = reconstruct

    def __repr__(self):
        return (f"<HyperbolicScheme flux_function={self.flux_function}, "
                f"limiter={self.limiter}, reconstruct={self.reconstruct}>")

