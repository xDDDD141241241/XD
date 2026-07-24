# Risk Monitor

One page that answers a single question every evening: **how unusual are market
conditions right now, and did anything change today?**

Everything we had scattered across separate tools — VIX term structure, VVIX,
MOVE, credit spreads, breadth, concentration, stock/bond correlation — is
gathered here and reduced to one number plus the reasons behind it.

---

## Setting it up (about ten minutes, once)

You do not need to read any of the code.

1. Make a new GitHub repository. Call it whatever you like. Set it to **public**
   — that is what makes the free web page work.
2. Upload these files, keeping the folder structure:

   ```
   risk_monitor.py
   selftest.py
   docs/index.html
   docs/data.json
   .github/workflows/risk_monitor.yml     <- rename risk_monitor_workflow.yml to this
   ```

3. In the repo, go to **Settings → Pages**. Under "Source" pick *Deploy from a
   branch*, choose branch `main` and folder `/docs`. Save.
4. Go to the **Actions** tab, click *Daily Risk Monitor*, then *Run workflow*.
   Give it two or three minutes.
5. Your page is live at `https://<your-username>.github.io/<repo-name>/`.
   Bookmark it on your phone.

From then on it updates itself every weekday at 00:30 German time, about half an
hour after the US close. You never touch it again.

The `data.json` that ships with this is **sample data** so the page is not blank
when you first open it. The page says so at the bottom until your first real run
replaces it.

---

## How to read the page

**The big number, 0 to 100.** Not a probability and not a forecast. It says how
unusual today is compared with the last three years. Every gauge is ranked
against its own history and then averaged, so a VIX of 22 and a credit spread of
340bp can be compared on the same footing.

- under 35 — calm
- 35 to 60 — elevated, worth a look, not a reaction
- 60 to 80 — several gauges stretched at once, this is when position size matters
- over 80 — broad simultaneous stress, the tail of the distribution

**The bars underneath.** Each gauge on the same 0–100 track, loudest at the top.
Far right means today is near the most extreme reading that gauge has produced
in three years. The value at the right of each row is the actual number; the
small grey figure below it is the ranking. The shape of the stack tells you
*what kind* of stress this is — all the volatility bars lit but credit quiet is a
very different situation from the reverse.

**Regime, in small type under the score.** Deliberately separate from the number.
The score measures stress level; regime measures trend health. They are different
questions and merging them hides information.

**Signals.** Yes/no conditions, not opinions. Green dot = historically a buying
condition, red = a warning. One firing alone is noise. Several at once, or a
signal that agrees with a high score, is the thing to notice.

---

## What is actually in it

Thirteen gauges across five families:

- **Volatility term structure** — VIX/VIX3M, VIX9D/VIX. Above 1.00 means near-term
  fear is priced above three-month fear, the classic risk-off trigger.
- **Volatility of volatility** — VIX, VVIX, VVIX/VIX, MOVE. Rises when people
  start paying up for tail protection and when the bond market gets nervous.
- **Credit** — high-yield spread level and its 20-day change. Credit is usually
  early to price real trouble.
- **Breadth** — advance/decline z-score, share of the index above its 20-day
  average, new highs minus new lows. Built directly from the 500 constituents,
  so no paid data feed is needed.
- **Structure** — cap-weight versus equal-weight over 63 days, and stock/bond
  correlation. The first tells you whether fewer names are carrying the index;
  the second tells you whether bonds are still cushioning equities.

If a data source fails, its weight is redistributed across the ones that
answered rather than quietly dragging the score toward zero. The page tells you
when that has happened.

---

## Checking it still works

Run `python risk_monitor.py --selftest`. It needs no internet. It builds a fake
calm market and a fake crisis and checks that the crisis actually scores higher,
that a broken data feed does not corrupt the score, and that low breadth reads
as *high* danger rather than low. It prints PASSED or FAILED.

---

## Honest limits

- The score is descriptive. High readings cluster around bad periods, but they
  also occur without incident. It tells you when to check your position sizing,
  not when to sell.
- Three years of ranking history means the 2020 and 2022 extremes gradually drop
  out of the comparison window. Readings will drift higher over a long calm
  stretch simply because the yardstick shrinks.
- Yahoo occasionally returns gaps in `^VIX3M` and `^VVIX`. The monitor tolerates
  this, but if the warning banner appears several days running, something
  upstream has broken.
- Everything here was tested against synthetic data only, because the machine it
  was written on has no market-data access. The first live run is the real test.
