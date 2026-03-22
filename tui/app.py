"""Main Textual Application for the autoresearch dashboard (Gaudi 3)."""

import os
import subprocess
import sys
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer

from tui.hardware import get_hardware_summary
from tui.parser import OutputParser, StepMetrics
from tui.widgets import (
    TrainingPanel, HardwarePanel, ExperimentsTable,
    ExperimentStatusPanel, ActivityLog,
)


def _get_process_rss_mb(pid: int) -> float:
    """Get RSS of a process in MB via ps command."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip()) / 1024
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


class DashboardApp(App):
    """Autoresearch training dashboard for Intel Gaudi 3."""

    TITLE = "autoresearch"
    SUB_TITLE = "Gaudi 3 HPU Training Dashboard"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Dark/Light"),
        ("r", "reload_experiments", "Reload"),
    ]

    def __init__(
        self,
        training_script: str = "train_gaudi.py",
        mode: str = "single",
        max_experiments: int = 100,
        run_tag: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._training_script = training_script
        self._mode = mode
        self._max_experiments = max_experiments
        self._run_tag = run_tag
        self._hw_info = get_hardware_summary()
        self._proc: subprocess.Popen | None = None
        self._parser = OutputParser()
        self._reader_thread: threading.Thread | None = None
        self._orchestrator = None
        self._memory_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            with Vertical(id="training-panel") as v:
                v.border_title = "Training"
                yield TrainingPanel(id="training")
            with Vertical(id="hardware-panel") as v:
                v.border_title = "Hardware"
                yield HardwarePanel(self._hw_info, id="hardware")
        if self._mode == "agent":
            with Vertical(id="experiment-status-panel") as v:
                v.border_title = "Experiment Loop"
                yield ExperimentStatusPanel(id="exp-status")
        with Vertical(id="experiments-panel") as v:
            v.border_title = "Experiments"
            yield ExperimentsTable(id="experiments")
        with Vertical(id="activity-panel") as v:
            v.border_title = "Activity Log"
            yield ActivityLog(id="activity")
        yield Footer()

    async def on_mount(self) -> None:
        log = self.query_one("#activity", ActivityLog)
        training = self.query_one("#training", TrainingPanel)

        log.log_message(f"Dashboard started -- {self._hw_info.get('chip_name', 'Unknown')}")
        log.log_message(f"Mode: {self._mode}")

        if self._mode == "watch":
            training.set_description("Watch mode -- no training")
            return
        if self._mode == "agent":
            log.log_message(f"Max experiments: {self._max_experiments}")
            self._start_orchestrator()
            return

        log.log_message(f"Training script: {self._training_script}")
        if not os.path.exists(self._training_script):
            log.log_message(f"Script not found: {self._training_script}", style="bold red")
            training.set_description(f"Error: {self._training_script} not found")
            return
        self._start_training()

    def _start_training(self) -> None:
        log = self.query_one("#activity", ActivityLog)
        training = self.query_one("#training", TrainingPanel)
        python = sys.executable
        cmd = [python, "-u", self._training_script]
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        log.log_message(f"Launching: {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, bufsize=0,
            )
        except Exception as e:
            log.log_message(f"Failed to start: {e}", style="bold red")
            training.set_description(f"Error: {e}")
            return
        log.log_message("Process started, reading output...")
        training.set_description("Compiling model (first step may take 30-60s)...")
        self._start_memory_polling()
        self._reader_thread = threading.Thread(
            target=self._reader_worker, args=(self._proc,), daemon=True)
        self._reader_thread.start()

    def _reader_worker(self, proc):
        buffer = ""
        try:
            while True:
                byte = proc.stdout.read(1)
                if not byte:
                    break
                char = byte.decode('utf-8', errors='replace')
                if char in ('\n', '\r'):
                    if buffer.strip():
                        self.call_from_thread(self._on_training_output, buffer)
                    buffer = ""
                else:
                    buffer += char
        except Exception:
            pass
        finally:
            if buffer.strip():
                self.call_from_thread(self._on_training_output, buffer)
            self.call_from_thread(self._on_training_done, proc.wait())

    def _on_training_output(self, line):
        training = self.query_one("#training", TrainingPanel)
        hardware = self.query_one("#hardware", HardwarePanel)
        log = self.query_one("#activity", ActivityLog)
        results = self._parser.parse_line(line)
        for item in results:
            if isinstance(item, StepMetrics):
                training.update_metrics(item)
            elif isinstance(item, str):
                log.log_message(item)
                if item.startswith("Backend:"):
                    backend = item.split("(")[0].replace("Backend:", "").strip()
                    training.set_backend(backend)
                if "peak_vram" in item and self._parser.final:
                    hardware.update_vram(self._parser.final.peak_vram_mb)
                    training.update_final(self._parser.final)

    def _on_training_done(self, returncode):
        log = self.query_one("#activity", ActivityLog)
        self._proc = None
        self._stop_memory_polling()
        if returncode == 0:
            log.log_message("Training process exited successfully.", style="bold green")
        else:
            log.log_message(f"Training process exited with code {returncode}.", style="bold red")
        self.action_reload_experiments()

    def _start_orchestrator(self):
        from tui.orchestrator import ExperimentOrchestrator, OrchestratorCallbacks
        log = self.query_one("#activity", ActivityLog)
        training = self.query_one("#training", TrainingPanel)
        exp_status = self.query_one("#exp-status", ExperimentStatusPanel)

        def on_status(status, message):
            self.call_from_thread(exp_status.update_status, status, message)
            self.call_from_thread(log.log_message, f"[{status.upper()}] {message}")

        def on_experiment_start(exp_num, desc, reasoning):
            self.call_from_thread(exp_status.set_experiment_info, exp_num, desc, reasoning)
            self.call_from_thread(log.log_message, f"Exp {exp_num}: {desc}", "bold cyan")
            self.call_from_thread(training.set_description, f"Exp {exp_num}: {desc}")
            self.call_from_thread(self._reset_training_panel)

        def on_training_output(line):
            self.call_from_thread(self._on_orchestrator_training_output, line)

        def on_experiment_complete(result):
            status_style = {
                "keep": "bold green", "discard": "bold red",
                "crash": "bold red", "baseline": "bold cyan",
            }.get(result.status, "white")
            msg = f"Result: {result.status.upper()} -- val_bpb={result.val_bpb:.4f}" if result.val_bpb > 0 else f"Result: {result.status.upper()}"
            self.call_from_thread(log.log_message, msg, status_style)
            self.call_from_thread(self.action_reload_experiments)

        def on_stats_update(total, kept, discarded, best_bpb):
            self.call_from_thread(exp_status.update_stats, total, kept, discarded, best_bpb)

        def on_error(message):
            self.call_from_thread(log.log_message, f"ERROR: {message}", "bold red")

        callbacks = OrchestratorCallbacks(
            on_status_change=on_status, on_experiment_start=on_experiment_start,
            on_training_output=on_training_output, on_experiment_complete=on_experiment_complete,
            on_stats_update=on_stats_update, on_error=on_error,
        )
        self._orchestrator = ExperimentOrchestrator(
            training_script=self._training_script,
            max_experiments=self._max_experiments, run_tag=self._run_tag,
            callbacks=callbacks,
        )
        log.log_message("Starting autonomous experiment loop...")
        self._start_memory_polling()
        self._orchestrator.start()

    def _reset_training_panel(self):
        training = self.query_one("#training", TrainingPanel)
        training._metrics = None
        training._final = None
        training._refresh_content()
        hardware = self.query_one("#hardware", HardwarePanel)
        hardware.reset_memory()
        self._parser = OutputParser()

    def _on_orchestrator_training_output(self, line):
        training = self.query_one("#training", TrainingPanel)
        hardware = self.query_one("#hardware", HardwarePanel)
        results = self._parser.parse_line(line)
        for item in results:
            if isinstance(item, StepMetrics):
                training.update_metrics(item)
            elif isinstance(item, str):
                if any(kw in item for kw in ("Backend:", "peak_vram", "val_bpb")):
                    log = self.query_one("#activity", ActivityLog)
                    log.log_message(item)
                if item.startswith("Backend:"):
                    backend = item.split("(")[0].replace("Backend:", "").strip()
                    training.set_backend(backend)
                if "peak_vram" in item and self._parser.final:
                    hardware.update_vram(self._parser.final.peak_vram_mb)
                    training.update_final(self._parser.final)

    def _start_memory_polling(self):
        self._stop_memory_polling()
        self._memory_timer = self.set_interval(2.0, self._poll_memory)

    def _stop_memory_polling(self):
        if self._memory_timer is not None:
            self._memory_timer.stop()
            self._memory_timer = None

    def _poll_memory(self):
        pid = None
        if self._proc and self._proc.returncode is None:
            pid = self._proc.pid
        if pid is None and self._orchestrator:
            orch_proc = getattr(self._orchestrator, '_proc', None)
            if orch_proc and orch_proc.returncode is None:
                pid = orch_proc.pid
        if pid:
            rss_mb = _get_process_rss_mb(pid)
            if rss_mb > 0:
                hardware = self.query_one("#hardware", HardwarePanel)
                hardware.update_live_memory(rss_mb)

    def action_reload_experiments(self):
        table = self.query_one("#experiments", ExperimentsTable)
        table.load_data()

    async def _on_exit(self):
        if self._orchestrator:
            self._orchestrator.stop()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
