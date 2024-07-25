import pytest
import subprocess


# def test_example_marimo():
#     from . import example_marimo as notebook
#     notebook.app.run()

def test_broken_notebook():
    from . import broken_notebook as notebook
    notebook.app.run()