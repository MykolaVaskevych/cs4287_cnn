import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():

    print('hello world')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
