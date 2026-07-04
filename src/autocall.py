"""
Autocallable notes: Athena and Phoenix.

Engine design
-------------
Observations are discrete (autocall trigger, coupon barrier) and the capital
protection barrier is European (observed at maturity only) — the market
standard for modern Athena/Phoenix issues. This means the underlying only
needs to be sampled *at observation dates*, which GBM allows exactly
(no discretization error, no fine time grid).

The simulation is split into two stages:

1. `simulate_events` — one Monte Carlo pass that records, per path, the
   redemption date, the capital redeemed, and the *number* of coupon units
   paid at each date. None of these depend on the coupon level.

2. Pricing. Because events are coupon-independent, the PV is exactly linear
   in the coupon:   price(c) = A + B·c
   with A = PV of capital redemption and B = PV of one coupon point across
   all paid coupon units. The fair coupon for a target issue price is then
   closed-form:      c* = (target − A) / B
   — no root-finding, no extra simulations.
"""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

ATHENA = "athena"
PHOENIX = "phoenix"


@dataclass(frozen=True)
class AutocallSpec:
    """Product terms. Levels are fractions of initial spot (1.0 = 100%)."""
    style: str = ATHENA            # "athena" or "phoenix"
    maturity: float = 5.0          # years
    n_obs: int = 5                 # number of observation dates (last = maturity)
    ac_trigger: float = 1.0        # autocall trigger at first observation
    stepdown: float = 0.0          # trigger decrease per period (fraction, e.g. 0.01)
    coupon_barrier: float = 0.70   # Phoenix only
    protection: float = 0.60       # capital barrier, European (at maturity)
    memory: bool = True            # Phoenix only: missed coupons recovered

    def obs_times(self) -> np.ndarray:
        return np.linspace(self.maturity / self.n_obs, self.maturity, self.n_obs)

    def trigger_levels(self) -> np.ndarray:
        i = np.arange(self.n_obs)
        return self.ac_trigger - i * self.stepdown


@dataclass
class AutocallEvents:
    """Coupon-independent simulation results (everything per 100 notional)."""
    spec: AutocallSpec
    obs_times: np.ndarray        # (n_obs,)
    df: np.ndarray               # (n_obs,) discount factors at obs dates
    red_idx: np.ndarray          # (n_paths,) index of redemption date
    capital: np.ndarray          # (n_paths,) capital redeemed, fraction of notional
    units: np.ndarray            # (n_paths, n_obs) coupon units paid at each date
    autocalled: np.ndarray       # (n_paths,) bool, redeemed early via trigger
    final_ratio: np.ndarray      # (n_paths,) S_T / S0 (only meaningful if not autocalled)
    A: float                     # PV of capital leg, per 100 notional
    B: float                     # PV of coupon leg per 1 coupon point

    @property
    def n_paths(self) -> int:
        return len(self.red_idx)

    def price(self, coupon: float) -> float:
        """Fair value per 100 notional for a coupon in points per period."""
        return self.A + self.B * coupon

    def per_path_pv(self, coupon: float) -> np.ndarray:
        df_red = self.df[self.red_idx]
        return 100.0 * self.capital * df_red + coupon * (self.units * self.df).sum(axis=1)

    def stderr(self, coupon: float) -> float:
        pv = self.per_path_pv(coupon)
        return float(pv.std(ddof=1) / np.sqrt(self.n_paths))

    def solve_coupon(self, target: float = 100.0) -> float:
        """Coupon (points per period) such that price == target. Exact, see module docstring."""
        if self.B <= 1e-12:
            return float("nan")
        return (target - self.A) / self.B

    # ----- risk / distribution analytics -----

    def redemption_distribution(self) -> dict:
        """Probability of each outcome: autocall at date i, redeemed at maturity, capital loss."""
        n = self.spec.n_obs
        p_ac = np.zeros(n)
        for i in range(n):
            p_ac[i] = np.mean((self.red_idx == i) & self.autocalled)
        at_maturity = ~self.autocalled
        p_loss = float(np.mean(at_maturity & (self.capital < 1.0 - 1e-12)))
        p_redeemed_mat = float(np.mean(at_maturity)) - p_loss
        return {"p_autocall_by_date": p_ac, "p_redeemed_at_maturity": p_redeemed_mat,
                "p_capital_loss": p_loss}

    def expected_life(self) -> float:
        return float(self.obs_times[self.red_idx].mean())

    def loss_stats(self) -> dict:
        lost = (~self.autocalled) & (self.capital < 1.0 - 1e-12)
        if not lost.any():
            return {"p_loss": 0.0, "avg_loss": 0.0, "worst_loss": 0.0}
        losses = 1.0 - self.capital[lost]
        return {"p_loss": float(lost.mean()),
                "avg_loss": float(losses.mean()) * 100.0,
                "worst_loss": float(losses.max()) * 100.0}


def _sample_at_obs(S0, vol, r, obs_times, n_paths, seed) -> np.ndarray:
    """Exact GBM sampling at observation dates. Returns (n_paths, n_obs)."""
    rng = np.random.default_rng(seed)
    dts = np.diff(np.concatenate([[0.0], obs_times]))
    Z = rng.standard_normal((n_paths, len(obs_times)))
    log_inc = (r - 0.5 * vol ** 2) * dts + vol * np.sqrt(dts) * Z
    return S0 * np.exp(np.cumsum(log_inc, axis=1))


def simulate_events(spec: AutocallSpec, S0, vol, r,
                    n_paths=50_000, seed=42, S_ref=None) -> AutocallEvents:
    """S_ref is the initial fixing that barriers/triggers refer to (defaults to
    S0). Pass the unbumped spot when computing spot Greeks, so that absolute
    barrier levels stay fixed while the diffusion starts from the bumped spot —
    otherwise the product re-strikes and price is homogeneous of degree 0 in S.
    """
    if S_ref is None:
        S_ref = S0
    obs_t = spec.obs_times()
    df = np.exp(-r * obs_t)
    S = _sample_at_obs(S0, vol, r, obs_t, n_paths, seed)
    ratio = S / S_ref
    n = spec.n_obs
    triggers = spec.trigger_levels()

    above_trigger = ratio >= triggers[None, :]

    # Redemption: first observation (before maturity) at/above the trigger.
    early = above_trigger[:, :-1]
    if early.shape[1] > 0:
        any_early = early.any(axis=1)
        red_idx = np.where(any_early, early.argmax(axis=1), n - 1)
    else:  # single observation = maturity, no early redemption possible
        any_early = np.zeros(n_paths, dtype=bool)
        red_idx = np.full(n_paths, n - 1)
    autocalled = any_early.copy()
    # At maturity, finishing above the trigger also counts as a (final) autocall
    # for Athena coupon purposes, but redemption date is maturity either way.
    final_above_trigger = above_trigger[:, -1] & ~any_early

    final_ratio = ratio[:, -1]
    capital = np.ones(n_paths)
    at_maturity = ~any_early
    protected = final_ratio >= spec.protection
    capital[at_maturity & ~protected] = final_ratio[at_maturity & ~protected]

    # "alive" mask: path still running at date i (i <= red_idx)
    alive = np.arange(n)[None, :] <= red_idx[:, None]

    units = np.zeros((n_paths, n), dtype=np.float64)
    if spec.style == ATHENA:
        # Snowball: (i+1) coupons paid at redemption if redeemed via trigger
        # (early autocall, or finishing above the final trigger at maturity).
        gets_coupons = any_early | final_above_trigger
        rows = np.where(gets_coupons)[0]
        units[rows, red_idx[rows]] = red_idx[rows] + 1.0
    elif spec.style == PHOENIX:
        above_cb = (ratio >= spec.coupon_barrier) & alive
        if spec.memory:
            missed = np.zeros(n_paths)
            for i in range(n):
                pay = above_cb[:, i]
                units[pay, i] = 1.0 + missed[pay]
                missed[pay] = 0.0
                live = alive[:, i]
                missed[live & ~pay] += 1.0
        else:
            units[above_cb] = 1.0
    else:
        raise ValueError(f"Unknown autocall style: {spec.style}")

    A = float(100.0 * (capital * df[red_idx]).mean())
    B = float((units * df).sum(axis=1).mean())

    return AutocallEvents(spec=spec, obs_times=obs_t, df=df, red_idx=red_idx,
                          capital=capital, units=units, autocalled=autocalled,
                          final_ratio=final_ratio, A=A, B=B)


def autocall_greeks(spec: AutocallSpec, S0, vol, r, coupon,
                    n_paths=50_000, seed=42) -> dict:
    """Finite-difference Greeks, common random numbers (same seed across bumps).

    Conventions (price in % of notional):
      delta : pts of notional per +1% spot move
      gamma : change of delta per +1% spot move
      vega  : pts per +1 vol point
      theta : pts per −1 calendar day
      rho   : pts per +1 rate point (100bp)
    """
    def px(S=S0, v=vol, rr=r, sp=spec):
        return simulate_events(sp, S, v, rr, n_paths, seed, S_ref=S0).price(coupon)

    base = px()
    h = 0.01 * S0
    up, down = px(S=S0 + h), px(S=S0 - h)
    delta = (up - down) / 2.0
    gamma = up - 2 * base + down

    vega = px(v=vol + 0.01) - base

    dt = 1.0 / 365.0
    if spec.maturity > dt * 2:
        spec_t = AutocallSpec(style=spec.style, maturity=spec.maturity - dt,
                              n_obs=spec.n_obs, ac_trigger=spec.ac_trigger,
                              stepdown=spec.stepdown, coupon_barrier=spec.coupon_barrier,
                              protection=spec.protection, memory=spec.memory)
        theta = px(sp=spec_t) - base
    else:
        theta = 0.0

    rho = px(rr=r + 0.01) - base
    return {"price": base, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}


def simulate_display_paths(spec: AutocallSpec, S0, vol, r,
                           n_display=120, sub_steps=12, seed=7):
    """Fine-grained sample paths for visualization, classified by fate.

    Returns (t_grid, paths, red_idx, fate) where fate is:
      i in [0, n_obs-2]  -> autocalled at observation i
      -1                 -> redeemed at maturity (capital protected)
      -2                 -> capital loss at maturity
    Paths are truncated (NaN) after their redemption date.
    """
    obs_t = spec.obs_times()
    n = spec.n_obs
    t_grid = np.concatenate([[0.0]] + [
        np.linspace(0 if i == 0 else obs_t[i - 1], obs_t[i], sub_steps + 1)[1:]
        for i in range(n)
    ])
    rng = np.random.default_rng(seed)
    dts = np.diff(t_grid)
    Z = rng.standard_normal((n_display, len(dts)))
    log_inc = (r - 0.5 * vol ** 2) * dts + vol * np.sqrt(dts) * Z
    paths = S0 * np.exp(np.concatenate(
        [np.zeros((n_display, 1)), np.cumsum(log_inc, axis=1)], axis=1))

    obs_cols = np.array([1 + i * sub_steps + (sub_steps - 1) for i in range(n)])
    ratio_obs = paths[:, obs_cols] / S0
    triggers = spec.trigger_levels()
    above = ratio_obs >= triggers[None, :]

    early = above[:, :-1]
    if early.shape[1] > 0:
        any_early = early.any(axis=1)
        red_idx = np.where(any_early, early.argmax(axis=1), n - 1)
    else:
        any_early = np.zeros(n_display, dtype=bool)
        red_idx = np.full(n_display, n - 1)

    fate = np.full(n_display, -1, dtype=int)
    fate[any_early] = red_idx[any_early]
    lost = ~any_early & (ratio_obs[:, -1] < spec.protection)
    fate[lost] = -2

    # truncate after redemption
    for p in range(n_display):
        cut = obs_cols[red_idx[p]]
        paths[p, cut + 1:] = np.nan

    return t_grid, paths, red_idx, fate
