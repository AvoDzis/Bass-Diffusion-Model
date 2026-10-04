# Bass Diffusion Model

A small Python module that fits the [Bass diffusion model](https://en.wikipedia.org/wiki/Bass_diffusion_model) to a product's sales series and plots the fitted adoption curve. I wrote it in May 2022.

The Bass model describes how a new product spreads through a market. It has three parameters:

- **p**, the coefficient of innovation (people who adopt on their own)
- **q**, the coefficient of imitation (people who adopt because others did)
- **m**, the market potential (total number of eventual adopters)

## What's in the repo

| Path | What it is |
|---|---|
| `BassModel/Bass.py` | Base class that defines the interface: `fit`, `predict`, `plot`, `plot_cdf`, `summary` |
| `BassModel/Bass_PolReg.py` | Fits the cumulative Bass curve with non-linear least squares (`scipy.optimize.curve_fit`, Levenberg–Marquardt) |
| `BassModel/Bass_LSE.py` | The classic OLS approach: regresses per-period sales on cumulative sales and its square (`statsmodels`), then solves for m, p and q |
| `data.csv` | Sample data: cumulative PlayStation 4 sales in millions of units, Aug 2014 – Jul 2021 |

## How to run

Requires Python 3 and the packages in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run this from the repo root:

```python
import BassModel

model = BassModel.Bass_PolReg('data.csv')
p, q, m = model.fit()  # innovation, imitation, market potential
model.predict()
model.summary()        # RMSE, R², and p/q/m with ±1 standard error
model.plot()           # fitted curve against the actual data points
```

On the sample data, `Bass_PolReg` estimates m ≈ 114 million units, p ≈ 0.086 and q ≈ 0.27, with R² ≈ 0.97.

The input file is a CSV with no header. The second column holds the sales values. The first column (a date label) is ignored.

## Limitations

- **The time axis is the row number, not the date.** The sample data has uneven gaps between points (from 1 month up to 23 months), so the fitted p and q describe "per row", not "per month".
- **The two classes expect different inputs.** `Bass_PolReg` fits the *cumulative* curve, which matches `data.csv`. `Bass_LSE` expects *per-period* sales and takes the cumulative sum itself, so on `data.csv` its estimates are meaningless (m ≈ 3,759). Use it only with per-period sales.
- `Bass_PolReg.plot_cdf()` takes the cumulative sum of a curve that is already cumulative, so its y-axis has no real meaning.
- There are no tests, and this isn't a packaged library (no `pip install`).

## References

- Bass, F. M. (1969). *A New Product Growth Model for Consumer Durables.* Management Science, 15(5), 215–227.
- Code that helped me while writing this:
  [alejandropuerto/product-market-forecasting-bass-model](https://github.com/alejandropuerto/product-market-forecasting-bass-model),
  [Fahad021/Bass-Difussion-Modell-with-python](https://github.com/Fahad021/Bass-Difussion-Modell-with-python),
  [NForouzandehmehr/Bass-Diffusion-model-for-short-life-cycle-products-sales-prediction](https://github.com/NForouzandehmehr/Bass-Diffusion-model-for-short-life-cycle-products-sales-prediction)

## License

No license. All rights reserved.
