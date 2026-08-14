"""
Plain-language reading guide for every gauge.

Kept separate from the maths on purpose: the wording gets rewritten far more
often than the calculations do, and mixing them meant every prose tweak risked
touching a formula. Four fields per gauge, in the order you actually use them
at a glance:

  plain   - what this measures, in everyday words
  compare - what to look at alongside it. A gauge read alone is half the story
  worry   - the specific combination that means something
  ignore  - the reading that looks alarming and is not. This field prevents
            more bad decisions than the other three combined
"""

GUIDE = {
 # ---- volatility ----------------------------------------------------------
 "vix": dict(
  plain="The size of the swing the market expects over the next month. The fear gauge.",
  compare="The index itself. Normally the two move in opposite directions.",
  worry="It rises on a day the index ALSO rises. That pairing is rare and usually means someone is positioning for trouble.",
  ignore="A spike during an obvious selloff. That is the market reacting, not warning."),

 "vvix": dict(
  plain="How jumpy the price of insurance itself is — the volatility of volatility.",
  compare="The VIX line just beside it.",
  worry="This rises while VIX stays flat. Protection is being bought before anything shows up in price.",
  ignore="Both rising together during a selloff. Entirely normal."),

 "vvix_vix": dict(
  plain="How expensive crash protection is relative to everyday nervousness.",
  compare="The VIX level, always. Never read this one on its own.",
  worry="A high ratio while VIX is low — a calm market where tails are being priced up. Historically a complacent setup.",
  ignore="The number by itself. It rises mechanically whenever VIX is very low, so a high reading can simply mean 'VIX is cheap'."),

 "move": dict(
  plain="The VIX of the bond market: expected swings in US Treasuries.",
  compare="VIX. Which of the two moves first tells you where the trouble is coming from.",
  worry="This leads VIX higher. The shock is coming from rates or policy, which plays out differently from an equity scare.",
  ignore="Small daily moves. Bonds are noisy."),

 "ts_vix_vix3m": dict(
  plain="One-month fear divided by three-month fear. Below 1.00 is the normal state.",
  compare="Where the index is trading — near a high, or already falling.",
  worry="It crosses above 1.00 while the index is STILL near a high. Traders are paying more for protection now than for later.",
  ignore="Above 1.00 during a selloff already underway. That is usually the panic, and often near the low."),

 "ts_vix9d_vix": dict(
  plain="Nine-day fear versus one-month fear. Picks up specific dated events.",
  compare="The calendar, not the chart.",
  worry="Rising while price is flat — something specific is being priced in, like a Fed meeting or a big earnings date.",
  ignore="Movement during a broad selloff. It is just following VIX."),

 "vol_floor": dict(
  plain="The calmest VIX of the last three months, against the same figure six months ago. Is the market's quiet baseline drifting upward?",
  compare="The index. Ideally the floor falls while price rises.",
  worry="The floor rising while the index makes new highs. The market can no longer get as quiet as it used to — a classic late-cycle tell.",
  ignore="A rising floor during a correction. That is simply what corrections do."),

 # ---- credit --------------------------------------------------------------
 "hy_oas": dict(
  plain="The extra yield lenders demand to lend to risky companies. Low means lenders are relaxed.",
  compare="The index, every time it prints a new high.",
  worry="Rising while stocks make highs. Lenders are seeing something equity holders are not.",
  ignore="The absolute level on its own — it drifts with the interest-rate environment."),

 "hy_oas_chg": dict(
  plain="Whether that lending premium has widened or narrowed over the past month.",
  compare="The index's own last month.",
  worry="Positive — widening — while the index makes new highs. This is the single most reliable disagreement signal in the whole panel.",
  ignore="Widening during a selloff. Everything widens then."),

 "quality_spread": dict(
  plain="The gap between what the weakest borrowers pay and what the merely mediocre ones pay.",
  compare="The headline high-yield spread directly above it.",
  worry="This widens while the headline spread stays tight. Trouble starts at the bottom of the credit stack and works upward.",
  ignore="A move driven by one sector. A few distressed energy or retail names can open this gap without meaning anything market-wide."),

 # ---- breadth -------------------------------------------------------------
 "breadth_ad_z": dict(
  plain="How strong day-to-day participation has been recently, measured in standard deviations.",
  compare="Nothing. Read it alone, as a strength reading.",
  worry="Sustained negative. But for a genuine divergence use the confirmation gap instead — this gauge does not know where price is.",
  ignore="Small swings. It is a noisy series."),

 "pct_above_20": dict(
  plain="What share of the 500 companies are trading above their own 20-day average. Very short-horizon.",
  compare="Whether the index has actually broken anything, or merely dipped.",
  worry="Under 15%. Though that is usually washout near a LOW, not a warning of one.",
  ignore="Anything between roughly 40 and 70. That is an ordinary market."),

 "nh_nl": dict(
  plain="How many companies hit a new 52-week high today, minus how many hit a new low.",
  compare="The index when it is at a new high.",
  worry="Negative while the index prints a new high. This was present at 1972, 2000, 2007 and 2021.",
  ignore="Daily noise. The cumulative version below is the more reliable form of the same idea."),

 "nhnl_cum_z": dict(
  plain="A running total of the line above, scored against its own past year. The classic breadth line.",
  compare="The index against its 200-day average.",
  worry="Below zero. That has marked the bear phases: 2001-03, 2008-09, 2015-16, 2022.",
  ignore="It as a top warning. It turned down well AFTER the 2000 and 2007 highs. It tells you whether dips are worth buying, not when to sell."),

 # ---- divergence ----------------------------------------------------------
 "ad_confirmation": dict(
  plain="Where the breadth line sits within its own year, minus where the index sits within its own. Are the two agreeing?",
  compare="Built in — the comparison IS the gauge.",
  worry="Clearly negative. The index is high in its range while participation is not: textbook non-confirmation.",
  ignore="Readings near zero. Breadth is keeping up, which is exactly what you want."),

 "breadth_divergence": dict(
  plain="Of the last 63 trading days, how many had the index within 1% of a high while breadth was already negative.",
  compare="Built in.",
  worry="A count climbing off zero and staying there. That is the November 2021 shape.",
  ignore="One or two days. Zero is healthy, single digits are a flicker."),

 "credit_divergence": dict(
  plain="The same count, for credit: days near a high while lending spreads were widening.",
  compare="Built in.",
  worry="A rising count. This one led 2000, 2007, 2018 and 2021-22.",
  ignore="Very little. Of all the divergences here this has the best record, so treat a sustained rise seriously."),

 # ---- structure -----------------------------------------------------------
 "concentration": dict(
  plain="How far the big-company index has outrun the equal-weighted version over three months.",
  compare="The index itself.",
  worry="Both rising together. The gain is coming from a handful of names rather than the market.",
  ignore="Negative readings. Those mean the average stock is keeping up, which is healthy."),

 "corr_spy_tlt": dict(
  plain="Whether stocks and Treasuries have been moving together or in opposite directions over three months.",
  compare="Your own hedges — not the index.",
  worry="Above zero. Bonds are no longer cushioning equity losses, so a drawdown costs more than the standard models assume.",
  ignore="It as a timing signal. It tells you what a fall would cost you, never whether one is coming."),

 "defensive_rotation": dict(
  plain="Utilities and consumer staples measured against the whole index over three months.",
  compare="The index.",
  worry="Defensives outperforming while the index makes highs. Money is moving to safety inside the rally.",
  ignore="It as an early warning. It confirms far more often than it leads."),

 "risk_appetite": dict(
  plain="Stocks and gold versus the dollar, the yen and the Swiss franc. The mood of money across asset classes.",
  compare="The index.",
  worry="Falling while stocks hold up. Capital is quietly moving into safe currencies — this is the only gauge here that can see outside the stock market.",
  ignore="It for now. Newly added and untested. Judge it on the turning-point section before giving it weight."),

 # ---- liquidity -----------------------------------------------------------
 "nfci": dict(
  plain="A broad Chicago Fed measure of how easy it is to borrow anywhere in the system. Above zero means tighter than normal.",
  compare="The index.",
  worry="Tightening while stocks sit at highs.",
  ignore="Day-to-day changes. It updates weekly, so treat it as background rather than a trigger."),

 "net_liquidity": dict(
  plain="Spare cash in the banking system: Fed holdings, minus the Treasury's account, minus money parked overnight.",
  compare="The index.",
  worry="Falling while the index makes highs. The advance is running on positioning rather than on new money.",
  ignore="It as a standalone trigger. The relationship is contested and the history is short."),
}
