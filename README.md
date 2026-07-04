# Structured Products Desk

Interactive pricing & analytics workbench for structured products, built with Python/Streamlit. Two workspaces:

- **Brick Builder** — compose payoffs from primitive legs (bonds, vanilla options, digitals, barrier options) and analyze them in real time: payoff diagrams, Black-Scholes / Monte Carlo pricing, Greeks, spot × vol heatmaps.
- **Autocall Pricer** — full pricing and risk workspace for **Athena** and **Phoenix** autocallable notes: fair-coupon solver, outcome-colored Monte Carlo paths, redemption probabilities, P&L distribution with VaR, and an auto-generated term sheet.

Built as a learning tool — not a production pricer. Assumes flat volatility, GBM dynamics, no dividends, no repo, no credit/funding spread.

---

## Highlights

### Autocall engine (`src/autocall.py`)

- **Exact sampling at observation dates.** Autocall triggers and coupon barriers are observed discretely, and the capital protection barrier is European (maturity only) — the market standard for modern issues. GBM can therefore be sampled *exactly* at the observation dates: no discretization error and no fine time grid (a 5Y annual note needs only 5 samples per path).
- **Price is linear in the coupon.** One Monte Carlo pass records, per path, the redemption date, the capital redeemed, and the *number* of coupon units paid — none of which depend on the coupon level. The PV is therefore exactly `price(c) = A + B·c`, and the **fair coupon solver** is closed-form: `c* = (100 − A) / B`. No root-finding, no extra simulation — the way a desk pricer back-solves the coupon to par.
- **Products covered:**
  - *Athena* — snowball coupons paid only at autocall (or at maturity above the final trigger), optional **step-down** trigger schedule.
  - *Phoenix* — conditional coupons below a separate coupon barrier, with or without **memory** (missed coupons recovered).
- **Greeks** by finite differences with common random numbers; spot bumps keep absolute barrier levels fixed (the product does not re-strike), which is what makes autocall delta non-trivial.

### Analytics

- Monte Carlo path visualization colored by outcome (autocalled early / redeemed at maturity / capital loss) with trigger diamonds and barrier lines
- Redemption distribution per observation date, expected life, P(capital loss), expected loss given loss
- Distribution of discounted P&L with VaR 95%
- Term sheet generated from live parameters (observation schedule with real dates, mechanism, model risk metrics), downloadable as HTML

### Builder

- 13 classic presets: Capital Protected Note, Reverse Convertible, Bonus / Discount Certificate, Straddle, Strangle, Call/Put Spread, Butterfly, Collar, Risk Reversal, barrier options
- Payoff decomposition per leg with strike/barrier annotations
- Greeks (closed-form for vanilla, finite-difference for MC legs), delta profile, spot × vol heatmap
- Put-call parity sanity check

---

## Install & run

```bash
cd structured-products-builder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project structure

```
structured-products-builder/
├── app.py                  Streamlit application (both workspaces)
├── requirements.txt
├── .streamlit/
│   └── config.toml         Theme configuration
├── assets/
│   └── style.css           Desk-pricer dark theme
└── src/
    ├── pricing.py          Black-Scholes closed-form + Monte Carlo barrier engine
    ├── instruments.py      Leg classes (Bond, Call, Put, Digital, Barrier)
    ├── portfolio.py        Portfolio composition and aggregation
    ├── presets.py          Built-in preset structures
    └── autocall.py         Athena / Phoenix engine: events MC, coupon solver, Greeks
```

---

## Model assumptions (honest list)

- **Flat volatility.** No smile/skew — OTM barriers, digitals and autocall coupons will deviate from market quotes (often materially: the down-and-in put embedded in an autocall is a skew product).
- **No dividends, no repo.** Underlying drifts at the risk-free rate.
- **Protection barrier is European** (observed at maturity), the modern standard; continuously-monitored barriers in the Brick Builder use 252 steps/year.
- **Single underlying.** No worst-of baskets (the dominant real-world autocall format).
- **Monte Carlo noise.** Standard error is displayed with the fair value; increase MC paths for tighter estimates. Fixed seed (42) for reproducibility.

---

## Possible extensions

- Volatility smile/skew (the natural next step — would materially reprice the autocalls)
- Worst-of baskets via Cholesky-correlated GBM
- Dividend yield / repo
- Issuer funding spread on the bond leg
- Side-by-side product comparison
