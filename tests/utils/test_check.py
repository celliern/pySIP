from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pysip.utils.check import check_model


def diagnostic_inputs():
    model = SimpleNamespace(ss=SimpleNamespace(A=np.diag([-1.0, -2.0])))
    summary = pd.DataFrame(
        {"|g(η)|": [1e-6, 1e-6], "|dpen(θ)|": [0.0, 0.0], "pvalue": [0.01, 0.02]}
    )
    return model, summary, np.eye(2)


def test_passing_diagnostics_do_not_raise():
    assert check_model(*diagnostic_inputs(), raise_error=True)


def test_failing_diagnostics_raise_only_failed_test():
    model, summary, corr = diagnostic_inputs()
    summary.loc[0, "pvalue"] = 0.5
    assert not check_model(model, summary, corr)
    with pytest.raises(RuntimeError, match="^Failed summary_pvalue$"):
        check_model(model, summary, corr, raise_error=True)
