"""module to crudely fit pydivest parameters to historical econic and resource
use data."""

import numpy as np
import sympy as sp
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from rs_models.econ_model import econ_model as em
DATA_PATH = '../data/'


class ParameterFit:
    """fit parameters to data and return them in dict"""

    def __init__(self, alpha=2./3., gamma=1./8., chi=0.02, s=0.25):
        """save parameter values and load data

        Parameters
        ----------
        alpha: float
            labor elasticity in production function. also defines capital
            elasticity in production function via alpha + beta = 1
        gamma: float
            elasticity of knowledge in clean sector production function
        chi: float
            depreciation rate of knowledge stock
        """

        self.alpha = alpha
        self.gamma = gamma
        self.chi = chi
        self.s = s

        self.datasets = {}

        self._load_labor_data()
        self._load_gdp_data()
        self._load_energy_data()
        self._load_oil_price_data()

        self.fitted_parameters = {}

    def _load_labor_data(self):
        """load labor data and add it to datasets"""

        labor = pd.read_csv(DATA_PATH +
                            'Labor/API_SL_TLF_TOTL_IN_DS2_en_csv_v2_59582.csv',
                            header=2)
        labor = labor.drop([
            'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 63'
        ],
                           axis=1)
        labor.set_index('Country Name', inplace=True)
        labor = labor.transpose().astype(float)
        labor.drop(list(labor.index.values[0:5]), inplace=True)
        labor['Total'] = labor.sum(axis=1) / 10.
        labor.index = labor.index.astype(int)

        self.datasets['labor'] = labor

    def _load_energy_data(self):
        """load energy data and add it to datasets"""

        energy = pd.read_excel(DATA_PATH + 'Energy_Consumption/' +
                               'bp-stats-review-2019-all-data.xlsx',
                               sheet_name=1)
        energy.set_index('In Mtoe', inplace=True)
        energy = energy.dropna().transpose()

        self.datasets['energy'] = energy

    def _load_gdp_data(self):
        """load gdp data and add it to datasets"""

        gdp = pd.read_csv(DATA_PATH +
                          'GDP/API_NY_GDP_MKTP_CD_DS2_en_csv_v2_40924.csv',
                          header=2)
        gdp = gdp.drop([
            'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 63'
        ],
                       axis=1)
        gdp.set_index('Country Name', inplace=True)
        gdp = gdp.transpose().astype(float)
        gdp.drop(list(gdp.index.values[0:5]), inplace=True)
        gdp['Total'] = gdp.sum(axis=1)
        gdp.index = gdp.index.astype(int)

        self.datasets['gdp'] = gdp

    def _load_oil_price_data(self):
        """load oil price data and add it to datasets"""

        # load historical oil prices to approximate historical fossil resource
        # cost.

        # MWV. (July 25, 2019). Average annual OPEC crude oil price from 1960
        # to 2019 (in U.S. dollars per barrel) [Chart].
        # In Statista. Retrieved July 25, 2019,
        # from https://www.statista.com/statistics/262858/change-in-opec-crude-oil-prices-since-1960/

        oil_price = pd.read_excel(
            DATA_PATH + 'Oil_Price/' +
            'statistic_id262858_opec-oil-price-annually-1960-2019.xlsx',
            sheet_name='Data',
            header=4)
        oil_price.index = oil_price.index.droplevel(0).astype(int)
        oil_price.head()

        # convert to price per ton
        toe_to_barrel = 7.1428571428571
        oil_price['Average price in U.S. dollars per ton'] = \
            oil_price['Average price in U.S. dollars per barrel'] \
            * toe_to_barrel

        self.datasets['oil price'] = oil_price

    def fit_params(self):
        """fit parameters to data given the input parameters provided
        """

        energy = self.datasets['energy']
        labor = self.datasets['labor']
        gdp = self.datasets['gdp']
        oil_price = self.datasets['oil price']
        alpha = self.alpha
        betac = 1-self.alpha
        betad = betac
        gamma = self.gamma
        chi = self.chi
        s = self.s

        data = energy.loc[:,['Fraction Fossil', 'Fraction Renewable', 'Total Fossil', 'Total Renewable']]
        data['labor'] = labor['Total']
        data['production dirty'] = gdp['Total'] * energy['Fraction Fossil']
        data['production clean'] = gdp['Total'] * energy['Fraction Renewable']

        # interpolate labor data
        dl = data.loc[data['labor']>0]
        itp = interp1d(dl.index.values, dl['labor'].values, fill_value='extrapolate')
        data['labor'] = [float(itp(x)) for x in data.index.values]

        # calculate energy intensity
        data['energy intensity'] = data['production dirty'] / data['Total Fossil']

        # capital income ratio 2010
        cir2010 = 4.4

        # capital in each sector as production in each sector times capital income ratio.

        clean_capital_2010 = cir2010 * data['production clean'].loc[2010]
        dirty_capital_2010 = cir2010 * data['production dirty'].loc[2010]

        # set capital depreciation such, that the capital stock is in equilibrium given the current income and savings rate.

        # \dot{K} = s * income - delta * K == 0
        # e.g. delta = s * income/K = s/capital income ratio

        delta = s/cir2010

        # calculate C according to 
        # \dot{C} = Yc - chi * C, chi = 0.02

        chi = self.chi
        C = 0

        for Yc in data.loc[range(1965,2011), 'production clean']:
            C += Yc - chi * C

        # Estimate current and initial resource stock from fossil resource usage.

        fossil_data = data.loc[data['Total Fossil']>0, 'Total Fossil']

        # cumulative historical usage
        R_cum = 0
        R_cum_t = []
        for R in list(fossil_data.values):
            R_cum += R
            R_cum_t.append(R_cum)

        # total stock estimated as cumulative historical usage plus another 100 years of current usage:
        G1 = 100*fossil_data.values[-1]
        G0 = R_cum + G1

        # timeseries of fossil resource data as initial resource minus cumulative resource usage
        data['Fossil Resource'] = (G0 - R_cum_t)

        # so, this rough estimate says, that we have used about one third of the total available amount of fossil fuels. Fair enough.

        # calculate approx total energy cost as price per ton * total fossil use per year in tons of oil equivalent
        data['Fossil resource cost data'] = oil_price['Average price in U.S. dollars per ton'] * data['Total Fossil'] * 10e6
        data['dirty production minus resource cost'] = data['production dirty'] - data['Fossil resource cost data']

        from scipy.optimize import least_squares

        def model(*args, **kwargs):

            [bR, mu] = args[0]

            cRm = [bR * r * (g / kwargs['G0'])**mu for r, g in zip(kwargs['R'], kwargs['G'])]

            return [x1 - x2 for x1, x2 in zip(cRm, kwargs['cR'])]

        x0 = (10e15, -2)
        xlower = (0, -8)
        xupper = (10e18, -2)

        res = least_squares(model,
                            x0,
                            bounds=(xlower, xupper),
                            kwargs={'G0': G0,
                                    'G': list(data['Fossil Resource'].values),
                                    'R': list(data['Total Fossil'].values),
                                    'cR': list(data['Fossil resource cost data'])
                                   })

        data['Fossil resource cost fit'] = [res['x'][0] * r * (g / G0)**res['x'][1] for r, g in zip(list(data['Total Fossil'].values),
                                                                                              list(data['Fossil Resource'].values))]
        bR, mu = res['x']
        data.head()

        # implement production functions for Yc & Yd

        Yc = em.c_expressions[0][em.Yc]
        Yd = em.c_expressions[0][em.Yd]

        parameter_substitutions = {em.C: C,
                                   em.G0: G0,
                                   em.G: G1,
                                   em.bR: bR,
                                   em.mu: mu,
                                   em.Kc: clean_capital_2010,
                                   em.Kd: dirty_capital_2010,
                                   em.e: data.loc[2010, 'energy intensity'],
                                   em.alpha: alpha,
                                   em.betac: betac,
                                   em.betad: betac,
                                   em.gamma: gamma,
                                   em.R: data.loc[2010,'Total Fossil'],
                                   em.L: data.loc[2010, 'labor']}

        fYc = Yc.subs(parameter_substitutions)
        fYd = Yd.subs(parameter_substitutions)
        [fYc, fYd]

        from scipy.optimize import root
        def rfoo(x, Ycd, Ydd):
            psubs = {em.bc: x[0],
                     em.bd: x[1]}
            res = [sp.re(sp.Abs(fYc.subs(psubs) - Ycd).evalf()), sp.re(sp.Abs(fYd.subs(psubs) - Ydd).evalf())]
            return res

        x_start = (1, 1)

        res = root(rfoo, x_start, args=(data.loc[2010, 'production clean'], data.loc[2010, 'production dirty']))
        bc, bd = res['x']

        # rescale bc and bd such that they are independend of the initial values of Kc, Kd, L and C and also independent of the input elasticities.

        nbc = bc * data.loc[2010, 'labor']**alpha * clean_capital_2010**betac * C**gamma
        nbd = bd * data.loc[2010, 'labor']**alpha * dirty_capital_2010**betad

        fitted_parameters = {'b_c': bc,
                             'b_d': bd,
                             'b_r0': bR,
                             'mu': mu,
                             'e': data.loc[2010, 'energy intensity'],
                             'kappa_c': betac,
                             'kappa_d': betad,
                             'pi': alpha,
                             'xi': gamma,
                             'd_k': delta,
                             'd_c': chi,
                             's': s,
                             'G_0': G0,
                             'G': G1,
                             'C': C,
                             'K_c0': clean_capital_2010,
                             'K_d0': dirty_capital_2010,
                             'L': data.loc[2010, 'labor'],
                             'nbc': nbc,
                             'nbd': nbd}

        self.fitted_parameters = fitted_parameters

    def get_params(self):
        """return dictionary of fitted parameters and initial conditions
        """
        return self.fitted_parameters


if __name__ == '__main__':

    params = ParameterFit(chi=0.1, gamma=0.1)
    params.fit_params()
    print(params.get_params())
