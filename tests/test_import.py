import subprocess
import sys


def test_package_import_has_no_output_or_side_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import hallthruster"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
