from .instruments import (
    Underlying, ZeroCouponBond, EuropeanCall, EuropeanPut,
    DigitalCall, DigitalPut, BarrierOption
)


def capital_protected_note(spot=100.0, participation=1.0):
    return [
        ZeroCouponBond(quantity=1.0, notional=100.0),
        EuropeanCall(quantity=participation, strike=spot),
    ]


def reverse_convertible(spot=100.0, coupon=0.08, strike_put=90.0):
    return [
        ZeroCouponBond(quantity=1.0 + coupon, notional=100.0),
        EuropeanPut(quantity=-1.0, strike=strike_put),
    ]


def bonus_certificate(spot=100.0, bonus_level=120.0, barrier=75.0):
    return [
        Underlying(quantity=1.0),
        BarrierOption(quantity=1.0, strike=bonus_level, barrier=barrier,
                      option_type="put", barrier_type="down-and-out"),
    ]


def discount_certificate(spot=100.0, cap=110.0):
    return [
        Underlying(quantity=1.0),
        EuropeanCall(quantity=-1.0, strike=cap),
    ]


def straddle(spot=100.0):
    return [
        EuropeanCall(quantity=1.0, strike=spot),
        EuropeanPut(quantity=1.0, strike=spot),
    ]


def strangle(spot=100.0, width=0.1):
    return [
        EuropeanCall(quantity=1.0, strike=spot * (1 + width)),
        EuropeanPut(quantity=1.0, strike=spot * (1 - width)),
    ]


def call_spread(spot=100.0, width=0.1):
    return [
        EuropeanCall(quantity=1.0, strike=spot),
        EuropeanCall(quantity=-1.0, strike=spot * (1 + width)),
    ]


def put_spread(spot=100.0, width=0.1):
    return [
        EuropeanPut(quantity=1.0, strike=spot),
        EuropeanPut(quantity=-1.0, strike=spot * (1 - width)),
    ]


def butterfly(spot=100.0, width=0.1):
    return [
        EuropeanCall(quantity=1.0, strike=spot * (1 - width)),
        EuropeanCall(quantity=-2.0, strike=spot),
        EuropeanCall(quantity=1.0, strike=spot * (1 + width)),
    ]


def collar(spot=100.0, call_strike_pct=0.1, put_strike_pct=0.1):
    return [
        EuropeanCall(quantity=-1.0, strike=spot * (1 + call_strike_pct)),
        EuropeanPut(quantity=1.0, strike=spot * (1 - put_strike_pct)),
    ]


def risk_reversal(spot=100.0, width=0.1):
    return [
        EuropeanCall(quantity=1.0, strike=spot * (1 + width)),
        EuropeanPut(quantity=-1.0, strike=spot * (1 - width)),
    ]


def up_and_out_call(spot=100.0, barrier_pct=0.2):
    return [
        BarrierOption(quantity=1.0, strike=spot,
                      barrier=spot * (1 + barrier_pct),
                      option_type="call", barrier_type="up-and-out"),
    ]


def down_and_in_put(spot=100.0, barrier_pct=0.2):
    return [
        BarrierOption(quantity=1.0, strike=spot,
                      barrier=spot * (1 - barrier_pct),
                      option_type="put", barrier_type="down-and-in"),
    ]


PRESETS = {
    "— Select a preset —": None,
    "Capital Protected Note (100% + Call)": capital_protected_note,
    "Reverse Convertible (Bond − Put)": reverse_convertible,
    "Bonus Certificate (Call + DO Put)": bonus_certificate,
    "Discount Certificate (Call Spread synthetic)": discount_certificate,
    "Straddle (long vol)": straddle,
    "Strangle (long vol, wider)": strangle,
    "Call Spread (bullish directional)": call_spread,
    "Put Spread (bearish directional)": put_spread,
    "Butterfly (range play)": butterfly,
    "Collar (hedge long stock)": collar,
    "Risk Reversal (bullish skew)": risk_reversal,
    "Up-and-Out Call (barrier)": up_and_out_call,
    "Down-and-In Put (barrier)": down_and_in_put,
}


PRESET_DESCRIPTIONS = {
    "Capital Protected Note (100% + Call)":
        "Principal protected at maturity + participation to upside via a long call at current spot.",
    "Reverse Convertible (Bond − Put)":
        "High coupon bond that converts into stock if spot < strike at maturity. Investor is short a put.",
    "Bonus Certificate (Call + DO Put)":
        "Receives a bonus level if barrier is never touched; otherwise payoff follows underlying.",
    "Discount Certificate (Call Spread synthetic)":
        "Buy underlying at discount in exchange for capped upside (short call above cap).",
    "Straddle (long vol)":
        "Long call + long put at same strike. Profits from large move either direction.",
    "Strangle (long vol, wider)":
        "Long OTM call + long OTM put. Cheaper than straddle, needs bigger move.",
    "Call Spread (bullish directional)":
        "Long lower call + short higher call. Limited cost, limited upside.",
    "Put Spread (bearish directional)":
        "Long higher put + short lower put. Limited cost, limited downside gain.",
    "Butterfly (range play)":
        "Long 1 ITM + short 2 ATM + long 1 OTM. Profits if spot stays near ATM.",
    "Collar (hedge long stock)":
        "Short OTM call + long OTM put. Caps both upside and downside for a holder.",
    "Risk Reversal (bullish skew)":
        "Long OTM call + short OTM put. Zero-cost bullish structure often.",
    "Up-and-Out Call (barrier)":
        "Standard call that knocks out (worthless) if underlying touches barrier above.",
    "Down-and-In Put (barrier)":
        "Put that activates only if underlying touches barrier below. Very common in autocalls.",
}
