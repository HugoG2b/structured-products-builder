from functools import lru_cache

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, r, sigma, T):
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, r, sigma, T):
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_digital_call(S, K, r, sigma, T):
    if T <= 0:
        return 1.0 if S > K else 0.0
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return np.exp(-r * T) * norm.cdf(d2)


def bs_digital_put(S, K, r, sigma, T):
    if T <= 0:
        return 1.0 if S < K else 0.0
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return np.exp(-r * T) * norm.cdf(-d2)


def bs_call_greeks(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1)
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T) / 100.0
    theta = (-S * pdf * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def bs_put_greeks(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1) - 1.0
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T) / 100.0
    theta = (-S * pdf * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def simulate_gbm_paths(S0, r, sigma, T, n_paths=10000, n_steps=252, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal(size=(n_paths, n_steps))
    drift = (r - 0.5 * sigma ** 2) * dt
    diff = sigma * np.sqrt(dt)
    log_returns = drift + diff * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


# Cached: returns a scalar, all args hashable. Makes tab switches / heatmap
# re-renders instant once a (params) combination has been priced.
@lru_cache(maxsize=8192)
def mc_barrier_option(S0, K, B, r, sigma, T, option_type, barrier_type,
                      n_paths=10000, n_steps=252, rebate=0.0, seed=42):
    paths = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed)
    final = paths[:, -1]
    max_path = paths.max(axis=1)
    min_path = paths.min(axis=1)

    if barrier_type == "up-and-out":
        active = max_path < B
    elif barrier_type == "down-and-out":
        active = min_path > B
    elif barrier_type == "up-and-in":
        active = max_path >= B
    elif barrier_type == "down-and-in":
        active = min_path <= B
    else:
        raise ValueError(f"Unknown barrier type: {barrier_type}")

    if option_type == "call":
        intrinsic = np.maximum(final - K, 0.0)
    else:
        intrinsic = np.maximum(K - final, 0.0)

    payoff = np.where(active, intrinsic, rebate)
    return np.exp(-r * T) * payoff.mean()


def mc_greeks_bump(pricer_fn, S0, bump_rel=0.01, vol_bump_abs=0.01, r_bump_abs=0.0001, T_bump_abs=1/365):
    base = pricer_fn(S=S0)
    up = pricer_fn(S=S0 * (1 + bump_rel))
    down = pricer_fn(S=S0 * (1 - bump_rel))
    dS = S0 * bump_rel
    delta = (up - down) / (2 * dS)
    gamma = (up - 2 * base + down) / (dS ** 2)
    vega = (pricer_fn(sigma_bump=vol_bump_abs) - base) / (vol_bump_abs * 100)
    theta = (pricer_fn(T_bump=-T_bump_abs) - base)
    rho = (pricer_fn(r_bump=r_bump_abs) - base) / (r_bump_abs * 100)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
