from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from . import pricing


@dataclass
class Instrument:
    quantity: float = 1.0

    def describe(self) -> str:
        raise NotImplementedError

    def payoff(self, S_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def price(self, S, vol, r, T, n_paths=10000, n_steps=252) -> float:
        raise NotImplementedError

    def greeks(self, S, vol, r, T, n_paths=10000, n_steps=252) -> dict:
        raise NotImplementedError

    def is_path_dependent(self) -> bool:
        return False


@dataclass
class Underlying(Instrument):
    """
    Spot underlying held from t=0 to maturity.
    payoff(T) = S_T, price(t=0) = S, delta = 1, all other Greeks = 0.
    No dividends, no repo (consistent with the rest of the model).
    """

    def describe(self) -> str:
        sign = "+" if self.quantity >= 0 else "-"
        return f"{sign}{abs(self.quantity):.2f} × Underlying"

    def payoff(self, S_array):
        return self.quantity * S_array

    def price(self, S, vol, r, T, **kwargs):
        return self.quantity * S

    def greeks(self, S, vol, r, T, **kwargs):
        return {
            "delta": self.quantity * 1.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }


@dataclass
class ZeroCouponBond(Instrument):
    notional: float = 100.0

    def describe(self) -> str:
        sign = "+" if self.quantity >= 0 else "-"
        return f"{sign}{abs(self.quantity):.2f} × ZCB (notional {self.notional:.0f})"

    def payoff(self, S_array):
        return self.quantity * self.notional * np.ones_like(S_array)

    def price(self, S, vol, r, T, **kwargs):
        return self.quantity * self.notional * np.exp(-r * T)

    def greeks(self, S, vol, r, T, **kwargs):
        pv = self.price(S, vol, r, T)
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                "theta": (r * pv) / 365.0,
                "rho": -T * pv / 100.0}


@dataclass
class EuropeanCall(Instrument):
    strike: float = 100.0

    def describe(self):
        sign = "+" if self.quantity >= 0 else "-"
        return f"{sign}{abs(self.quantity):.2f} × European Call (K={self.strike:.2f})"

    def payoff(self, S_array):
        return self.quantity * np.maximum(S_array - self.strike, 0.0)

    def price(self, S, vol, r, T, **kwargs):
        return self.quantity * pricing.bs_call(S, self.strike, r, vol, T)

    def greeks(self, S, vol, r, T, **kwargs):
        g = pricing.bs_call_greeks(S, self.strike, r, vol, T)
        return {k: self.quantity * v for k, v in g.items()}


@dataclass
class EuropeanPut(Instrument):
    strike: float = 100.0

    def describe(self):
        sign = "+" if self.quantity >= 0 else "-"
        return f"{sign}{abs(self.quantity):.2f} × European Put (K={self.strike:.2f})"

    def payoff(self, S_array):
        return self.quantity * np.maximum(self.strike - S_array, 0.0)

    def price(self, S, vol, r, T, **kwargs):
        return self.quantity * pricing.bs_put(S, self.strike, r, vol, T)

    def greeks(self, S, vol, r, T, **kwargs):
        g = pricing.bs_put_greeks(S, self.strike, r, vol, T)
        return {k: self.quantity * v for k, v in g.items()}


@dataclass
class DigitalCall(Instrument):
    strike: float = 100.0
    cash: float = 1.0

    def describe(self):
        sign = "+" if self.quantity >= 0 else "-"
        return f"{sign}{abs(self.quantity):.2f} × Digital Call (K={self.strike:.2f}, pay={self.cash:.2f})"

    def payoff(self, S_array):
        return self.quantity * self.cash * (S_array > self.strike).astype(float)

    def price(self, S, vol, r, T, **kwargs):
        return self.quantity * self.cash * pricing.bs_digital_call(S, self.strike, r, vol, T)

    def greeks(self, S, vol, r, T, **kwargs):
        # Finite-difference Greeks (digital has no clean closed form for all Greeks)
        h = S * 0.001
        base = self.price(S, vol, r, T)
        delta = (self.price(S + h, vol, r, T) - self.price(S - h, vol, r, T)) / (2 * h)
        gamma = (self.price(S + h, vol, r, T) - 2 * base + self.price(S - h, vol, r, T)) / (h ** 2)
        vega = (self.price(S, vol + 0.01, r, T) - base) / 1.0
        theta = (self.price(S, vol, r, max(T - 1/365, 1e-6)) - base)
        rho = (self.price(S, vol, r + 0.0001, T) - base) / 0.01
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


@dataclass
class DigitalPut(Instrument):
    strike: float = 100.0
    cash: float = 1.0

    def describe(self):
        sign = "+" if self.quantity >= 0 else "-"
        return f"{sign}{abs(self.quantity):.2f} × Digital Put (K={self.strike:.2f}, pay={self.cash:.2f})"

    def payoff(self, S_array):
        return self.quantity * self.cash * (S_array < self.strike).astype(float)

    def price(self, S, vol, r, T, **kwargs):
        return self.quantity * self.cash * pricing.bs_digital_put(S, self.strike, r, vol, T)

    def greeks(self, S, vol, r, T, **kwargs):
        h = S * 0.001
        base = self.price(S, vol, r, T)
        delta = (self.price(S + h, vol, r, T) - self.price(S - h, vol, r, T)) / (2 * h)
        gamma = (self.price(S + h, vol, r, T) - 2 * base + self.price(S - h, vol, r, T)) / (h ** 2)
        vega = (self.price(S, vol + 0.01, r, T) - base) / 1.0
        theta = (self.price(S, vol, r, max(T - 1/365, 1e-6)) - base)
        rho = (self.price(S, vol, r + 0.0001, T) - base) / 0.01
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


@dataclass
class BarrierOption(Instrument):
    strike: float = 100.0
    barrier: float = 120.0
    option_type: str = "call"
    barrier_type: str = "up-and-out"
    rebate: float = 0.0

    def describe(self):
        sign = "+" if self.quantity >= 0 else "-"
        return (f"{sign}{abs(self.quantity):.2f} × Barrier {self.option_type.capitalize()} "
                f"({self.barrier_type}, K={self.strike:.2f}, B={self.barrier:.2f})")

    def is_path_dependent(self):
        return True

    def payoff(self, S_array):
        """
        Payoff at maturity, with deterministic knockout / activation regions handled.

        Convention: barrier_type prefix implies barrier position relative to current spot.
          - up-X    : barrier is ABOVE current spot
          - down-X  : barrier is BELOW current spot

        Deterministic regions (path continuity guarantees outcome):
          - up-and-out    : S_T >= B → must have crossed B → knocked out → 0
          - down-and-out  : S_T <= B → must have crossed B → knocked out → 0
          - up-and-in     : S_T >= B → must have crossed B → activated   → intrinsic
          - down-and-in   : S_T <= B → must have crossed B → activated   → intrinsic

        Ambiguous regions (path could have touched B and bounced back):
          - OUT options: show alive payoff (best case for holder)
          - IN  options: show 0 (conservative: assume not activated)

        Real expected value is computed via MC in price(); this is just for visualization.
        """
        if self.option_type == "call":
            intrinsic = np.maximum(S_array - self.strike, 0.0)
        else:
            intrinsic = np.maximum(self.strike - S_array, 0.0)

        if self.barrier_type == "up-and-out":
            alive = S_array < self.barrier
            intrinsic = np.where(alive, intrinsic, 0.0)
        elif self.barrier_type == "down-and-out":
            alive = S_array > self.barrier
            intrinsic = np.where(alive, intrinsic, 0.0)
        elif self.barrier_type == "up-and-in":
            activated = S_array >= self.barrier
            intrinsic = np.where(activated, intrinsic, 0.0)
        elif self.barrier_type == "down-and-in":
            activated = S_array <= self.barrier
            intrinsic = np.where(activated, intrinsic, 0.0)

        return self.quantity * intrinsic

    def price(self, S, vol, r, T, n_paths=10000, n_steps=252, **kwargs):
        return self.quantity * pricing.mc_barrier_option(
            S, self.strike, self.barrier, r, vol, T,
            self.option_type, self.barrier_type,
            n_paths=n_paths, n_steps=n_steps, rebate=self.rebate
        )

    def greeks(self, S, vol, r, T, n_paths=10000, n_steps=252, **kwargs):
        h = max(S * 0.01, 0.01)
        vh = 0.01
        base = self.price(S, vol, r, T, n_paths, n_steps)
        p_up = self.price(S + h, vol, r, T, n_paths, n_steps)
        p_down = self.price(S - h, vol, r, T, n_paths, n_steps)
        p_vup = self.price(S, vol + vh, r, T, n_paths, n_steps)
        p_tup = self.price(S, vol, r, max(T - 1/365, 1e-6), n_paths, n_steps)
        p_rup = self.price(S, vol, r + 0.0001, T, n_paths, n_steps)
        return {
            "delta": (p_up - p_down) / (2 * h),
            "gamma": (p_up - 2 * base + p_down) / (h ** 2),
            "vega": (p_vup - base),
            "theta": (p_tup - base),
            "rho": (p_rup - base) / 0.01,
        }
