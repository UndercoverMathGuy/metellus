"""Public API surface. End-users build an `Operations` graph and hand it
to `run()`; everything else (fusion, assembly, lifetime analysis,
scheduling, execution) is internal."""

from api.run import RunResult, run

__all__ = ["RunResult", "run"]
