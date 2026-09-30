import pytest


@pytest.mark.xfail(reason="Historical JSON deserialization remains incomplete in M1")
def test_json_runner_support_is_deferred():
    pytest.fail("JSON runner is not yet a supported public API")
