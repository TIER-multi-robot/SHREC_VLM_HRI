"""Sidecar call logging for new model runs.

Writes one JSON line per API call to SHREC/logs/calls/<run>.jsonl with:
timestamp, model, model_version, backend, precision, input/output tokens,
latency, truncation flag, sample index and error flag.

ER runs predate this logger; their missing fields are recorded as a
limitation and are not backfilled.
"""

import json
import os
import threading

_lock = threading.Lock()


def log_call(log_path, record):
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with _lock:
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


def call_logger_for(log_path, run_name, model, backend, sample_index_ref):
    """Return a callable(record_extra) that fills in run-level fields.

    sample_index_ref: dict with key 'i' updated by main_vlm_exp each iteration
    (Agent.chat is called with no sample argument in the existing code).
    """

    def log(**extra):
        record = {
            'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
            'run_name': run_name,
            'model': model,
            'backend': backend,
            'sample_index': sample_index_ref.get('i'),
        }
        record.update(extra)
        log_call(log_path, record)

    return log
