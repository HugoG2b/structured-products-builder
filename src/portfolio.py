from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd

from .instruments import Instrument


@dataclass
class Portfolio:
    instruments: List[Instrument] = field(default_factory=list)

    def add(self, instrument: Instrument):
        self.instruments.append(instrument)

    def remove(self, index: int):
        if 0 <= index < len(self.instruments):
            self.instruments.pop(index)

    def clear(self):
        self.instruments.clear()

    def payoff_at_maturity(self, S_array: np.ndarray) -> np.ndarray:
        if not self.instruments:
            return np.zeros_like(S_array)
        total = np.zeros_like(S_array, dtype=float)
        for inst in self.instruments:
            total += inst.payoff(S_array)
        return total

    def leg_payoffs(self, S_array: np.ndarray) -> dict:
        return {inst.describe(): inst.payoff(S_array) for inst in self.instruments}

    def price(self, S, vol, r, T, n_paths=10000, n_steps=252) -> float:
        return sum(inst.price(S, vol, r, T, n_paths=n_paths, n_steps=n_steps)
                   for inst in self.instruments)

    def breakdown(self, S, vol, r, T, n_paths=10000, n_steps=252) -> pd.DataFrame:
        rows = []
        for inst in self.instruments:
            rows.append({
                "Leg": inst.describe(),
                "Price": inst.price(S, vol, r, T, n_paths=n_paths, n_steps=n_steps),
                "Path-dependent": "Yes" if inst.is_path_dependent() else "No",
            })
        if rows:
            df = pd.DataFrame(rows)
            total = df["Price"].sum()
            df = pd.concat([df, pd.DataFrame([{"Leg": "TOTAL", "Price": total, "Path-dependent": "-"}])],
                           ignore_index=True)
            return df
        return pd.DataFrame(columns=["Leg", "Price", "Path-dependent"])

    def greeks(self, S, vol, r, T, n_paths=10000, n_steps=252) -> dict:
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for inst in self.instruments:
            g = inst.greeks(S, vol, r, T, n_paths=n_paths, n_steps=n_steps)
            for k in totals:
                totals[k] += g.get(k, 0.0)
        return totals

    def is_empty(self) -> bool:
        return len(self.instruments) == 0

    def has_path_dependent(self) -> bool:
        return any(inst.is_path_dependent() for inst in self.instruments)
