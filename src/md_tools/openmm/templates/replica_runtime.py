#!/usr/bin/env python
"""What a generated replica-exchange input imports. A facade, deliberately thin.

    from replica_runtime import REST2Protocol

    protocol = REST2Protocol(
        tau=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        temperature_k=300.0,
        timestep_fs=4.0,
        segment_ps=2.0,
        exchange_interval_ps=10.0,
        whole_output_interval_ps=10.0,
        number_of_exchanges=1000,
    )

That is the whole contract. The generated input names a protocol and nothing else: no path, no
exchange loop, no NetCDF, no MPI, no source parsing. The executor reads `protocol` out of the file
and runs it.
"""
from replica_protocol import KB_KJ_PER_MOL_K, ProtocolError, REST2Protocol   # noqa: F401

__all__ = ["REST2Protocol", "ProtocolError", "KB_KJ_PER_MOL_K"]
