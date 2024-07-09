import marimo

__generated_with = "0.7.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import polars as pl
    import numpy as np
    a = np.arange(20000).reshape(200,100)
    b = np.arange(100)
    df = pl.from_numpy(a, schema={"a":pl.Array(pl.Float64, 100)})
    df2=df.select(np.matmul(pl.col("a"),b))
    return a, b, df, df2, np, pl


@app.cell
def __(df2):
    df2
    return


@app.cell
def __(b, df, np, pl):
    df.select(pl.col("a").map_batches(np.matmul(pl.col("a"), b)))
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
