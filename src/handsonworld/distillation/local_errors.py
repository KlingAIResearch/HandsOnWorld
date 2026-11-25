"""Keep a node-local traceback even when shared output storage cannot write."""

from functools import wraps
import os
from pathlib import Path
import socket
import sys
import tempfile
import traceback


def record_local_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            # DCP CheckpointException inherits BaseException, not Exception.
            # Preserve help/normal exits without manufacturing an error log.
            if isinstance(error, SystemExit) and error.code in (None, 0):
                raise
            # Run inside elastic's @record, before writing shared error.json.
            try:
                kind = os.environ.get('FASTGEN_ERROR_KIND', 'ODE').upper()
                directory = Path(
                    os.environ.get('FASTGEN_ERROR_DIR')
                    or os.environ.get(f'{kind}_ERROR_DIR')
                    or f'/tmp/fastgen-{kind.lower()}-errors'
                )
                directory.mkdir(parents=True, exist_ok=True)
                prefix = f'{socket.gethostname()}-rank{os.environ.get("RANK", "0")}-pid{os.getpid()}-'
                with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', prefix=prefix, suffix='.log',
                    dir=directory, delete=False,
                ) as handle:
                    handle.write(traceback.format_exc())
                    handle.flush()
                    os.fsync(handle.fileno())
                print(f'{kind} local traceback: {handle.name}', file=sys.stderr, flush=True)
            except Exception:
                # Local storage or redirected stderr may also be unavailable.
                # Never replace the exception that stopped generation.
                pass
            raise

    return wrapped
