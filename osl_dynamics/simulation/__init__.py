"""Simulations for time series data.

"""

from osl_dynamics.simulation.base import Simulation
from osl_dynamics.simulation.sin import SingleSine
from osl_dynamics.simulation.mar import MAR
from osl_dynamics.simulation.mvn import MVN, MS_MVN
from osl_dynamics.simulation.hmm import (
    HMM,
    HMM_MAR,
    HMM_MVN,
    MS_HMM_MVN,
    HierarchicalHMM_MVN,
    HMM_Sine,
)
from osl_dynamics.simulation.hsmm import HSMM, HSMM_MVN, MixedHSMM_MVN
from osl_dynamics.simulation.sm import MixedSine
