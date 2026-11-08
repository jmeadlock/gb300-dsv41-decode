"""pytest plugin (M2-L5, tests/test_pdl_impl_cpu.py): at session end write the names of every kernel the shim launched
with launch_pdl=True to $PDL_COV_OUT (JSON). Load with `-p pdl_cov_plugin` from tests/."""
import json
import os


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("PDL_COV_OUT")
    if not path:
        return
    import triton_shim

    with open(path, "w") as f:
        json.dump({"pdl_kernels": sorted(triton_shim.KERNEL_NAMES_PDL), "exitstatus": int(exitstatus)}, f)
