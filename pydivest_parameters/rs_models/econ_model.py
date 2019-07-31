"""This is a simple two sector investment mode, that can be fitted to empirical
data"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from tqdm import tqdm

sp.init_printing()


class econ_model():

    # Define variables and parameters for the economic subsystem:

    L, Lc, Ld = sp.symbols('L L_c L_d', positive=True, real=True)

    K, Kc, Kd = sp.symbols('K K_c K_d', positive=True, real=True)
    w, rc, rd = sp.symbols('w r_c r_d', positive=True, real=True)
    R, G, C = sp.symbols('R, G, C', positive=True, real=True)
    eta, s, delta, chi, alpha, betac, betad, gamma = \
        sp.symbols('eta s delta chi alpha beta_c beta_d gamma',
                   positive=True, rational=True, real=True)
    mu = sp.symbols('mu', negative=True, rational=True, real=True)
    bc, bd, bR, e, G0 = sp.symbols('b_c b_d b_R e G_0',
                                   positive=True,
                                   real=True)
    Xc, Xd, XR = sp.symbols('X_c X_d X_R', positive=True, real=True)
    Yc, Yd, cR = sp.symbols('Y_c Y_d c_R', positive=True, real=True)
    dKc, dKd, dG, dC = sp.symbols(r'\dot{K}_c \dot{K}_d \dot{G} \dot{C}',
                                  real=True)

    # Solutions to the algebraic constraints from labor and capital markets

    s1 = {
        w: alpha * L**(alpha - 1) * (Xc + Xd * XR)**(1 - alpha),
        rc: betac / Kc * Xc * L**alpha * (Xc + Xd * XR)**(-alpha),
        rd: betad / Kd * Xd * XR * L**alpha * (Xc + Xd * XR)**(-alpha),
        R: bd / e * Kd**betad * L**alpha * (Xd * XR / (Xc + Xd * XR))**alpha,
        Lc: L * Xc / (Xc + Xd * XR),
        Ld: L * Xd * XR / (Xc + Xd * XR),
    }

    s2 = {
        Xc: (bc * Kc**betac * C**gamma)**(1. / (1 - alpha)),
        Xd: (bd * Kd**betad)**(1. / (1 - alpha)),
        XR: (1. - bR / e * (G / G0)**mu)**(1. / (1 - alpha))
    }

    outa_gas = bR / e * (G / G0)**mu

    c_outputs = {
        Yc: bc * Lc**alpha * Kc**betac * C**gamma,
        Yd: bd * Ld**alpha * Kd**betad,
        rc: rc,
        rd: rd,
        C: C,
        G: G
    }

    c_kdots = {
        dKc: s * eta * (w * L + rd * Kd + rc * Kc) - delta * Kc,
        dKd: s * (1 - eta) * (w * L + rd * Kd + rc * Kc) - delta * Kd,
        dG: -R,
        dC: bc * Lc**alpha * Kc**betac * C**gamma - chi * C
    }

    c_expressions = [c_outputs, c_kdots]

    for d in c_expressions:
        for key, exp in d.items():
            d[key] = exp.subs(s1).subs(s2)

    def __init__(self,
                 L: list,
                 R: list,
                 e: list,
                 bc=1,
                 bd=4,
                 alpha=2 / 3,
                 betac=1 / 3,
                 betad=1 / 3,
                 gamma=1 / 8,
                 mu=-2,
                 bR=1,
                 G0=1000,
                 s=0.25,
                 delta=0.4,
                 chi=0.2,
                 eta=0.02):

        self.parameter_substitutions = {
            self.bc: bc,
            self.bd: bd,
            self.alpha: alpha,
            self.betac: betac,
            self.betad: betad,
            self.gamma: gamma,
            self.mu: mu,
            self.bR: bR,
            self.G0: G0,
            self.s: s,
            self.delta: delta,
            self.chi: chi,
            self.eta: eta
        }
        self.kdots = {}
        self.outputs = {}
        self.expressions = [self.outputs, self.kdots]

        for c_d, d in zip(self.c_expressions, self.expressions):
            for key, exp in c_d.items():
                d[key] = sp.lambdify(
                    (self.L, self.Kc, self.Kd, self.C, self.R, self.G, self.e),
                    exp.subs(self.parameter_substitutions), 'numpy')

        # system time
        self.t = 0

        # energy intensity as a variable parameter
        self.te = e

        # trajectories from data
        self.tL = L
        self.tR = R

        # trajectories from the model
        self.tKc = []
        self.tKd = []
        self.tC = []
        self.tG = []

        self.t_outputs = []

        # break condition
        self.outa_gas = sp.lambdify(
            (self.L, self.Kc, self.Kd, self.C, self.R, self.G, self.e),
            self.outa_gas.subs(self.parameter_substitutions), 'numpy')

    def initial(self, Kc0, Kd0, C0, G0=None):
        """initialize model with capital and knowledge stock.
        resource stock optional.

        Parameters
        ----------
        Kc0: float
            initial capital stock in the clean sector
        Kd0: float
            initial capital stock in the dirty sector
        C0: float
            initial knowledge stock in the clean sector
        G0: float
            initial fossil resource stock
        """

        self.tKc += [Kc0]
        self.tKd += [Kd0]
        self.tC += [C0]

        if G0 is None:
            self.tG += [self.G0]
        else:
            self.tG += [G0]

    def _eval(self, L, Kc, Kd, C, R, G, e):
        """evaluates model equations for given variable values
        """

        variables = (L, Kc, Kd, C, R, G, e)

        if self.outa_gas(*variables) < 1:
            dots = {key: exp(*variables) for key, exp in self.kdots.items()}
            outputs = {
                key: exp(*variables)

                for key, exp in self.outputs.items()
            }
        else:
            raise ValueError('outa gas')

        return dots, outputs

    def _step(self):
        """integrates model one step forward in time
        """

        # set current value of energy intensity
        e_t = self.te[self.t]

        # set current value of variables
        l_t = self.tL[self.t]
        k_ct = self.tKc[self.t]
        k_dt = self.tKd[self.t]
        c_t = self.tC[self.t]
        r_t = self.tR[self.t]
        g_t = self.tG[self.t]

        # get time derivatives of variables
        dots, outputs = self._eval(L=l_t,
                                   Kc=k_ct,
                                   Kd=k_dt,
                                   C=c_t,
                                   R=g_t,
                                   G=g_t,
                                   e=e_t)

        # set next value of variables
        k_ctp1 = k_ct + dots[self.dKc]
        k_dtp1 = k_dt + dots[self.dKd]
        c_tp1 = c_t + dots[self.dC]
        g_tp1 = g_t - r_t

        # append to trajectories
        self.tKc += [k_ctp1]
        self.tKd += [k_dtp1]
        self.tC += [c_tp1]
        self.tG += [g_tp1]

        self.t_outputs += [[val for val in outputs.values()]]

        self.t += 1

    def run(self, t_max: int):
        """run model until t=t_max
        """

        for _ in range(t_max):
            self._step()

            if self.tG[-1] <= 0:
                return

    def get_trajectory(self):
        """return model trajectory
        """

        return pd.DataFrame(data=self.t_outputs,
                            columns=self.outputs.keys()).astype(float)


if __name__ == '__main__':
    model = econ_model(L=range(1, 101),
                       R=np.ones(101),
                       e=101 * [10],
                       bd=2,
                       eta=.1)
    model.initial(Kc0=1,
                  Kd0=100,
                  C0=1,
                  G0=model.parameter_substitutions[model.G0])
    model.run(t_max=100)
    trj = model.get_trajectory()
    trj[[model.Yc, model.Yd]].astype(float).plot()
    plt.show()
