"""A BespokeFit coordinator whose startup runs once, and whose HTTP calls have a long timeout.

Selected with BEFLOW_COORDINATOR_ROUTER=coordinator_once:router.

1. With the installed FastAPI the coordinator's startup handler runs twice, which starts two
   coordinator cycles; each submits every stage (every torsion scan, the fit) a second time.
   The duplicates have identical inputs, so the answer is unchanged and the cost is doubled. The
   handler is wrapped to run at most once per process.
2. The coordinator posts each stage with `httpx.AsyncClient()`, whose default timeout is 5 s. A
   large optimisation schema exceeds that and the stage fails with "ReadTimeout". The default here
   is $BESPOKE_HTTP_TIMEOUT seconds (600).

No installed file is modified.
"""
import functools
import os

import httpx

_AsyncClient = httpx.AsyncClient


class _LongTimeoutAsyncClient(_AsyncClient):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("timeout", httpx.Timeout(float(os.environ.get("BESPOKE_HTTP_TIMEOUT", "600"))))
        super().__init__(*args, **kwargs)


httpx.AsyncClient = _LongTimeoutAsyncClient

from openff.bespokefit.executor.services.coordinator import app as _coordinator_app  # noqa: E402

router = _coordinator_app.router


def _once(fn):
    done = {"ran": False}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if done["ran"]:
            return None
        done["ran"] = True
        return fn(*args, **kwargs)

    wrapper._once = True
    return wrapper


router.on_startup[:] = [f if getattr(f, "_once", False) else _once(f) for f in router.on_startup]
