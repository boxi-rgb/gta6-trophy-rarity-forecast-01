# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/boxi-rgb/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/boxi-rgb/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, one week after launch, as five quantiles per trophy,
for make.fun's [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Method

`main.py` (Python 3.13, standard library only) does not extrapolate the rate
directly. Exophase refreshes its counts on its own schedule, so the page is a
staircase — long plateaus, then a jump — and a rate's move is the tug of war
between earners growing and players-tracked growing. The program therefore
decomposes the rate into earners E and players tracked N:

- **Compress** the history into refresh states (consecutive identical reads
  carry no information), recovering E as `round(rate * N / 100)`.
- **Extrapolate** `ln N` and `ln E` against `ln(1 + hours since release)` by
  weighted least squares. Small-E trophies have their slope shrunk toward a
  pooled slope (tier-group pooled when states are few, fully pooled when
  there are two or fewer). A second E estimate from the latest increment
  ratio `dE/dN` is averaged in when refreshes exist.
- **Median** `r50 = E(target) / N(target)`. The story chain
  Ch.1 ≥ Ch.2 ≥ Ch.3 ≥ Ch.4 ≥ final mission ≥ 100% ≥ platinum is enforced on
  the medians; Gone Fishing (id 2) is pinned near 100% and the platinum is
  capped low — almost nobody legitimately platinums the game in week one.
- **Dilution prior:** when the history has seen no refresh at all (the
  practice cases), the measured first refresh (638 → 3,868 players tracked)
  is used as a prior so the forecast still dilutes instead of predicting a
  flat page.
- **Spread** is built in logit space from three components — a random-walk
  term, a refresh-jump term from the cross-trophy spread of log-rate jumps at
  past refreshes, and a fit-residual term — then calibrated by a
  rolling-origin backtest run inside the program on the case's own history.

The program is fully deterministic (no randomness, fixed order, 4-decimal
half-up rounding) and runs in well under a second per case.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory, in name order, and writes one
`<id>.out` per case to the output directory.

## Referral

Built with make.fun — referral code `cwqsp23g`.

## License

MIT. See [LICENSE](LICENSE).
