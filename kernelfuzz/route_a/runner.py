"""Isolated Route A execution with bounded artifact observation."""

import os
import json
import hashlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .model import Observation, TestCase
from .observer import classify_observation

TRACE_PREAMBLE = Path(__file__).with_name("tracing.py").read_text(encoding="utf-8")


def _uninterruptible(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'D'
    except (OSError, IndexError):
        return False


def _stop_worker(process, errors):
    """Bound termination/reaping even when SIGKILL is delayed by disk I/O."""
    try:
        returncode = process.poll()
        if returncode is not None:
            return returncode, True
    except Exception as exc:
        errors.append(f"poll: {type(exc).__name__}: {exc}")
    try:
        if _uninterruptible(process.pid):
            errors.append('kill_skipped: worker is in uninterruptible D state; defer reaping')
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # The worker can exit between poll() and killpg(). Still wait().
    except OSError as exc:
        errors.append(f"killpg: {type(exc).__name__}: {exc}")
        try:
            process.kill()
        except OSError as exc:
            errors.append(f"kill: {type(exc).__name__}: {exc}")
    try:
        return process.wait(timeout=10), True
    except subprocess.TimeoutExpired:
        errors.append("reap_timeout: worker did not exit within 10s after SIGKILL")
    except Exception as exc:
        errors.append(f"reap: {type(exc).__name__}: {exc}")
    return None, False


def _reap_later(process, case_dir):
    """Keep ownership of a delayed child until wait() can reap it."""
    def reap():
        while True:
            try:
                process.wait()
                break
            except Exception:
                # Do not abandon the child on a transient wait failure.
                time.sleep(1)
        shutil.rmtree(case_dir, ignore_errors=True)

    threading.Thread(target=reap, name=f"route-a-reap-{process.pid}", daemon=True).start()


def _read_log(path, errors):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        errors.append(f"read {path.name}: {type(exc).__name__}: {exc}")
        return ""


class RouteAExecutor:
    def __init__(self, python_executable: Optional[str] = None, timeout_seconds: int = 60,
                 work_dir: Optional[str] = None, runtime_root: Optional[str] = None,
                 worker_env: Optional[dict] = None, persistent_log_root: Optional[str] = None):
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.work_dir = Path(work_dir).resolve() if work_dir else None
        self.runtime_root = Path(runtime_root).resolve() if runtime_root else None
        self.worker_env = dict(worker_env or {})
        self.persistent_log_root = Path(persistent_log_root) if persistent_log_root else None

    @staticmethod
    def _script(testcase: TestCase) -> str:
        return (
            TRACE_PREAMBLE
            + "\ntry:\n"
            + "    exec(compile(" + repr(testcase.code) + ", '<route-a-program>', 'exec'))\n"
            + "finally:\n"
            + "    print('ROUTE_A_KERNEL_EVENTS=' + _ra_json.dumps(_route_a_events), flush=True)\n"
            + "    print('ROUTE_A_HOOK_STATUS=' + (','.join(_route_a_hook_status) or 'none'), flush=True)\n"
        )

    def run(self, testcase: TestCase, target_tokens: Iterable[str] = (),
            selected_target: Optional[str] = None) -> Observation:
        del target_tokens
        started = time.monotonic()
        base = self.runtime_root or Path(tempfile.gettempdir())
        case_id = getattr(testcase, "test_id", hashlib.sha256(testcase.code.encode()).hexdigest()[:24])
        case_dir = base / "cases" / case_id
        script_path = case_dir / "worker.py"
        stdout_path = case_dir / "stdout.log"
        stderr_path = case_dir / "stderr.log"
        if self.persistent_log_root:
            self.persistent_log_root.mkdir(parents=True, exist_ok=True)
            stdout_path = self.persistent_log_root / (case_id + '.stdout.log')
            stderr_path = self.persistent_log_root / (case_id + '.stderr.log')
        timed_out = False
        returncode = None
        process = None
        reaped = True
        errors = []
        worker_error = False
        try:
            try:
                case_dir.mkdir(parents=True, exist_ok=True)
                script_path.write_text(self._script(testcase), encoding="utf-8")
                env = os.environ.copy()
                env.update(self.worker_env)
                env.setdefault("PYTHONHASHSEED", "0")
                for key, name in (("TMPDIR", "tmp"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                                  ("TRITON_CACHE_DIR", "triton")):
                    directory = case_dir / name
                    directory.mkdir(parents=True, exist_ok=True)
                    env[key] = str(directory)
                env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
                with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
                        stderr_path.open("w", encoding="utf-8") as stderr_handle:
                    process = subprocess.Popen(
                        [self.python_executable, str(script_path)],
                        cwd=str(self.work_dir) if self.work_dir else None,
                        env=env,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        start_new_session=True,
                    )
                    (case_dir / 'worker.json').write_text(json.dumps({'pid':process.pid,
                        'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                            if Path(f'/proc/{process.pid}/stat').exists() else None,
                        'started':time.time(), 'stdout':str(stdout_path), 'stderr':str(stderr_path)}))
                    reaped = False
                    returncode = process.wait(timeout=self.timeout_seconds)
                    reaped = True
            except subprocess.TimeoutExpired:
                timed_out = True
            except Exception as exc:
                worker_error = True
                errors.append(f"worker: {type(exc).__name__}: {exc}")
            finally:
                if process is not None and not reaped:
                    returncode, reaped = _stop_worker(process, errors)
            stdout = _read_log(stdout_path, errors)
            stderr = _read_log(stderr_path, errors)
            if errors:
                stderr += "\nROUTE_A_WORKER_ERRORS=" + "\n".join(errors) + "\n"
            observation = classify_observation(
                returncode, timed_out, stdout, stderr,
                graph_manifest=getattr(testcase, "graph_manifest", None),
                selected_target=selected_target,
            )
            if worker_error and not timed_out:
                observation.status = "execution_failure"
                observation.status_reason = "worker_error"
                observation.failure_stage = "worker"
            observation.elapsed_seconds = time.monotonic() - started
            observation.worker_metadata = {'pid': process.pid if process is not None else None,
                'reaped': reaped, 'case_dir': str(case_dir), 'cleanup_deferred': not reaped,
                'stdout_path': str(stdout_path), 'stderr_path': str(stderr_path),
                'timed_out': timed_out, 'errors': errors}
            return observation
        finally:
            # The generated program and observation evidence are persisted by
            # EventStore. Per-case compiler caches are intentionally ephemeral.
            if reaped:
                shutil.rmtree(case_dir, ignore_errors=True)
            else:
                _reap_later(process, case_dir)
