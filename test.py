import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import beta
from scipy.special import beta as beta_point
import math

survival_perc = [100.0, 64.3, 50.2, 41.5, 35.6, 31.1, 28.2, 25.4, 23.1, 21.0, 19.5, 19.5, 17.9]

CAC = 100 # Customer Acquisition Cost
AOR = 1.8  # Average Order Rate
RPO = 57.3 # Revenue per Order
CM = .26 # Contribution Margin
M = AOR * RPO * CM # Total Margin
WACC = 0.0153 # Discount Rate

St_actual = list(map(lambda x : x / 100, survival_perc))

T = len(St_actual)

Rt_actual = [i / j for i, j in zip(St_actual[1:T], St_actual[0:(T-1)])]
Rt_actual

lsts = {"month": range(1, T + 1), "St_actual": St_actual, "Rt_actual": [None] + Rt_actual}

d = pd.DataFrame(lsts)

t_calib = 6

def BG_rt(gamma: float, delta: float, t: int) -> float: 
    return ((delta + t - 1) / (gamma + delta + t - 1))

def BG_rt_sse(params: list, d: pd.DataFrame) -> float:
    gamma = pow(params[0], math.e)
    delta = pow(params[1], math.e)

    t = d.index[1:]
    Rt_actual = d["Rt_actual"].iloc[1:]

    pred = BG_rt(gamma, delta, t)

    return sum(pow(Rt_actual - pred, 2))

# Defining point calculations for the beta geometric model
BG_churn = lambda gamma, delta, t: beta_point(gamma + 1, delta + t - 1) / beta_point(gamma, delta) 
BG_survival = lambda gamma, delta, t: beta_point(gamma, delta + t) / beta_point(gamma, delta)

# Defining More Robust Model Takeaways
def BG_rlv(gamma: float, delta: float, cf: float, disc: float, past_t: int, t_end: int = 2000) -> pd.DataFrame:
    """
    Generate valuation data from beta-geometric input statistics after a certian time t.

    :param gamma: Gamma statistic of the beta model
    :param delta: Delta statistics of the beta model
    :param cf: Per purchase cash flow
    :param disc: Discount rate for time-value calculations
    :param cac: Cost of acquiring a customer in t = 0
    :param t: The number of periods to predict out
    """
    rlv = pd.DataFrame({
        'cf': cf,
        'period': range(past_t + 1, t_end + 1),
        'period_lag' : range(past_t, t_end)
    })

    rlv["st"] = BG_survival(gamma, delta, rlv["period"])
    rlv["st_past_t"] = BG_survival(gamma, delta, past_t)
    rlv["st_conditional"] = rlv["st"] / rlv["st_past_t"]
    rlv["pt_conditional"] = [1 - rlv["st_conditional"].iloc[0], *((rlv["st_conditional"] - rlv["st_conditional"].shift(-1))[0:-1])]
    rlv["discounted_RL"] = [0, *((1 / pow(1 + disc, rlv["period_lag"] - past_t - 1))[1:])]
    rlv["discounted_cf"] = rlv["cf"] * rlv["discounted_RL"]
    rlv["rlv"] = round(np.cumsum(rlv["discounted_cf"]), 2)

    rlv_sum_nest = rlv.groupby("rlv").agg({
        "rlv": ["size"],
        'pt_conditional': ["sum"]
    })

    rlv_sum = pd.DataFrame({"rlv": rlv_sum_nest.index, "num_exact_values": rlv_sum_nest["rlv"]["size"], "pt_conditional_sum": rlv_sum_nest["pt_conditional"]["sum"]})

    print(rlv_sum["pt_conditional_sum"].iloc[0:-1])

    rlv_sum["pt_cond_conv"] = [*(rlv_sum["pt_conditional_sum"].iloc[0:-1]), 1 - sum(rlv_sum["pt_conditional_sum"].iloc[0:-1])]
    rlv_sum["erlv"] = sum(rlv_sum["rlv"] * rlv_sum["pt_cond_conv"])
    rlv_sum["past_t"] = past_t

    return rlv_sum

preds = minimize(BG_rt_sse, (1, 1), d[0:t_calib + 1])

gamma_nls = pow(preds["x"][0], math.e)
delta_nls = pow(preds["x"][1], math.e)

BG_rlv(gamma_nls, delta_nls, M, WACC, 4)