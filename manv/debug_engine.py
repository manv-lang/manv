
from __future__ import annotations

import ast as pyast
from dataclasses import dataclass, field
import io
from pathlib import Path
import threading
from typing import Any, Callable
from uuid import uuid4

from .compiler import analyze_program, parse_program
from .debug_mapping import ExecPoint, SourceMap, build_source_map_from_hlir, provenance_from_kernel_op
from .diagnostics import ManvError
from .graph_capture import GraphCaptureTracer
from .hlir import HModule, Provenance
from .hlir_interpreter import HLIRFrame, HLIRInterpreter, HLIRTracer
from .hlir_lowering import lower_ast_to_hlir
from .kernel_ir import lower_graph_to_kernel
from .kernel_mock import execute_kernel_ir
from .object_runtime import ExceptionObject, InstanceObject


SessionId = str
EventListener = Callable[[dict[str, Any]], None]


@dataclass
class Breakpoint:
    id: int
    source_uri: str
    line: int
    column: int | None = None
    condition: str | None = None
    hit_condition: str | None = None
    log_message: str | None = None
    verified: bool = False
    message: str | None = None
    point: ExecPoint | None = None
    hits: int = 0


@dataclass
class StepPlan:
    mode: str
    start_frame_depth: int
    start_point_key: str
    start_line: int


@dataclass
class LogicalFrame:
    id: int
    function: str
    block: str
    depth: int
    provenance: Provenance | None
    args: dict[str, Any]
    locals: dict[str, Any]
    temporaries: dict[str, Any]


@dataclass
class ScopeRef:
    id: int
    kind: str
    frame_id: int


@dataclass
class ValueRef:
    id: int
    value_kind: str
    backing_object: Any
    paging_info: dict[str, Any] | None = None


class _OutputSink(io.StringIO):
    def __init__(self, session: "_RuntimeSession", engine: "DebugEngine"):
        super().__init__()
        self.session = session
        self.engine = engine

    def write(self, s: str) -> int:
        n = super().write(s)
        if s:
            self.engine._emit_event(self.session, "output", {"category": "stdout", "output": s})
        return n


@dataclass
class _RuntimeSession:
    id: str
    module: HModule
    source_map: SourceMap
    source_uri: str
    stop_on_entry: bool
    trace_compare: bool
    state: str = "NotStarted"
    thread: threading.Thread | None = None
    interpreter: HLIRInterpreter | None = None
    tracer: "_DebugTracer" | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    cond: threading.Condition = field(init=False)
    listeners: list[EventListener] = field(default_factory=list)
    breakpoints_by_file: dict[str, list[Breakpoint]] = field(default_factory=dict)
    breakpoints_by_point: dict[str, list[Breakpoint]] = field(default_factory=dict)
    exception_filters: set[str] = field(default_factory=lambda: {"all"})
    stop_reason: str | None = None
    stop_message: str | None = None
    current_point: ExecPoint | None = None
    current_provenance: Provenance | None = None
    step_plan: StepPlan | None = None
    pause_requested: bool = False
    pending_entry_stop: bool = False
    terminate_requested: bool = False
    next_bp_id: int = 1
    frame_tokens: dict[int, LogicalFrame] = field(default_factory=dict)
    next_frame_id: int = 1
    scope_tokens: dict[int, tuple[str, int]] = field(default_factory=dict)
    next_scope_id: int = 1
    value_tokens: dict[int, ValueRef] = field(default_factory=dict)
    next_value_id: int = 1
    globals: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    generated_sources: dict[int, dict[str, str]] = field(default_factory=dict)
    next_source_ref: int = 1
    event_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cond = threading.Condition(self.lock)


class _DebugTracer(HLIRTracer):
    def __init__(self, engine: "DebugEngine", session: _RuntimeSession):
        self.engine = engine
        self.session = session
        self.interpreter: HLIRInterpreter | None = None

    def bind_interpreter(self, interpreter: HLIRInterpreter) -> None:
        self.interpreter = interpreter

    def before_instruction(self, frame: HLIRFrame, block_label: str, instr: Any, resolved_args: list[Any]) -> None:
        del resolved_args
        session = self.session
        point = ExecPoint(kind="HLIR", function=frame.function, block=block_label, op_id=instr.instr_id or "")
        provenance = session.source_map.bind(point)

        with session.cond:
            session.current_point = point
            session.current_provenance = provenance
            self.engine._refresh_frames_locked(session)

            if session.terminate_requested:
                raise RuntimeError("debug session terminated")

            if session.pending_entry_stop:
                session.pending_entry_stop = False
                self.engine._stop_locked(session, "entry", "stopOnEntry", point, provenance)

            bp = self.engine._matching_breakpoint_locked(session, point)
            if bp is not None:
                env = self.engine._current_eval_env_locked(session)
                if self.engine._breakpoint_should_stop(bp, env):
                    if bp.log_message:
                        self.engine._emit_event(session, "output", {"category": "console", "output": bp.log_message + "\n"})
                    else:
                        self.engine._stop_locked(session, "breakpoint", "breakpoint", point, provenance)

            if session.pause_requested:
                session.pause_requested = False
                self.engine._stop_locked(session, "pause", "paused", point, provenance)

            if session.step_plan is not None and self.engine._step_boundary_reached(session, point, provenance):
                self.engine._stop_locked(session, "step", session.step_plan.mode, point, provenance)

            while session.state == "Stopped" and not session.terminate_requested:
                session.cond.wait()

    def on_exception(self, frame: HLIRFrame, block_label: str, instr: Any, error: Exception) -> None:
        session = self.session
        point = ExecPoint(kind="HLIR", function=frame.function, block=block_label, op_id=instr.instr_id or "")
        provenance = session.source_map.bind(point)
        with session.cond:
            self.engine._refresh_frames_locked(session)
            if "all" in session.exception_filters:
                self.engine._stop_locked(session, "exception", str(error), point, provenance)
                while session.state == "Stopped" and not session.terminate_requested:
                    session.cond.wait()

    def on_gpu_execution(self, fn_name: str, details: dict[str, Any]) -> None:
        session = self.session
        kernel_names = [str(name) for name in details.get("kernel_names", []) if name]
        if not kernel_names:
            return

        source_body: dict[str, Any] = {}
        with session.cond:
            source_text = str(details.get("cuda_source", ""))
            if source_text:
                source_ref = self.engine._register_generated_source_locked(
                    session,
                    name=f"{kernel_names[0]}.cu",
                    text=source_text,
                )
                source_body = {"name": f"{kernel_names[0]}.cu", "sourceReference": source_ref}

        body: dict[str, Any] = {
            "category": "console",
            "output": f"Executed as GPU kernel: {', '.join(kernel_names)}\n",
        }
        if source_body:
            body["source"] = source_body
        self.engine._emit_event(session, "output", body)

class DebugEngine:
    """Protocol-agnostic debugger runtime that powers DAP requests."""

    _registry: dict[str, _RuntimeSession] = {}

    def __init__(self) -> None:
        self._sessions: dict[str, _RuntimeSession] = {}

    def launch(self, program: str, args: list[str] | None = None, options: dict[str, Any] | None = None) -> SessionId:
        del args  # reserved for future runtime argv plumbing
        opts = options or {}
        source_uri = str(Path(program).resolve())
        source = Path(source_uri).read_text(encoding="utf-8")

        parsed = parse_program(source, source_uri)
        analyze_program(parsed, source_uri)
        hlir_module = lower_ast_to_hlir(parsed, source_name=source_uri)
        source_map = build_source_map_from_hlir(hlir_module)

        sid = str(uuid4())
        session = _RuntimeSession(
            id=sid,
            module=hlir_module,
            source_map=source_map,
            source_uri=source_uri,
            stop_on_entry=bool(opts.get("stopOnEntry", False)),
            trace_compare=bool(opts.get("traceCompare", False)),
        )
        session.pending_entry_stop = session.stop_on_entry

        tracer = _DebugTracer(self, session)
        out_sink = _OutputSink(session=session, engine=self)
        interpreter = HLIRInterpreter(stdout=out_sink, tracer=tracer)
        tracer.bind_interpreter(interpreter)

        session.tracer = tracer
        session.interpreter = interpreter
        session.state = "Initializing"

        thread = threading.Thread(target=self._run_session, args=(session,), name=f"manv-dbg-{sid}", daemon=True)
        session.thread = thread

        self._sessions[sid] = session
        DebugEngine._registry[sid] = session
        thread.start()
        return sid

    def attach(self, session_or_pid: str, options: dict[str, Any] | None = None) -> SessionId:
        del options
        if session_or_pid in self._sessions:
            return session_or_pid
        existing = DebugEngine._registry.get(session_or_pid)
        if existing is None:
            raise RuntimeError(f"attach target not found: {session_or_pid}")
        self._sessions[existing.id] = existing
        return existing.id

    def add_listener(self, session_id: str, listener: EventListener) -> None:
        session = self._session(session_id)
        with session.lock:
            session.listeners.append(listener)
            # Runtime execution can start before DAP attaches its listener, so
            # every emitted event is replayable. Replaying the full history is
            # simpler and more deterministic than trying to reconstruct only the
            # "current state" event after the fact.
            history = list(session.event_history)
        for payload in history:
            try:
                listener(payload)
            except Exception:
                continue

    def remove_listener(self, session_id: str, listener: EventListener) -> None:
        session = self._session(session_id)
        with session.lock:
            if listener in session.listeners:
                session.listeners.remove(listener)

    def disconnect(self, session_id: str) -> None:
        self.terminate(session_id)

    def terminate(self, session_id: str) -> None:
        session = self._session(session_id)
        with session.cond:
            session.terminate_requested = True
            if session.state != "Terminated":
                session.state = "Terminated"
            session.cond.notify_all()
        self._emit_event(session, "terminated", {})

    def resume(self, session_id: str, thread_id: int = 1) -> None:
        del thread_id
        session = self._session(session_id)
        with session.cond:
            session.step_plan = None
            self._set_running_locked(session)

    def pause(self, session_id: str, thread_id: int = 1) -> None:
        del thread_id
        session = self._session(session_id)
        with session.cond:
            session.pause_requested = True

    def step_in(self, session_id: str, thread_id: int = 1) -> None:
        del thread_id
        session = self._session(session_id)
        with session.cond:
            self._set_step_plan_locked(session, "In")
            self._set_running_locked(session)

    def step_over(self, session_id: str, thread_id: int = 1) -> None:
        del thread_id
        session = self._session(session_id)
        with session.cond:
            self._set_step_plan_locked(session, "Over")
            self._set_running_locked(session)

    def step_out(self, session_id: str, thread_id: int = 1) -> None:
        del thread_id
        session = self._session(session_id)
        with session.cond:
            self._set_step_plan_locked(session, "Out")
            self._set_running_locked(session)

    def set_source_breakpoints(self, session_id: str, file_uri: str, items: list[dict[str, Any]]) -> list[Breakpoint]:
        session = self._session(session_id)
        resolved = str(Path(file_uri).resolve())
        with session.lock:
            for bp in session.breakpoints_by_file.get(resolved, []):
                if bp.point is not None:
                    key = bp.point.key()
                    bps = session.breakpoints_by_point.get(key, [])
                    session.breakpoints_by_point[key] = [x for x in bps if x.id != bp.id]

            bound: list[Breakpoint] = []
            session.breakpoints_by_file[resolved] = []

            for item in items:
                bp = Breakpoint(
                    id=session.next_bp_id,
                    source_uri=resolved,
                    line=int(item.get("line", 1)),
                    column=item.get("column"),
                    condition=item.get("condition"),
                    hit_condition=item.get("hitCondition"),
                    log_message=item.get("logMessage"),
                )
                session.next_bp_id += 1

                point = session.source_map.find_first_point(resolved, bp.line, bp.column)
                if point is None:
                    bp.verified = False
                    bp.message = "No executable statement at this location"
                else:
                    bp.point = point
                    bp.verified = True
                    key = point.key()
                    session.breakpoints_by_point.setdefault(key, []).append(bp)
                session.breakpoints_by_file[resolved].append(bp)
                bound.append(bp)

        for bp in bound:
            self._emit_event(
                session,
                "breakpoint",
                {
                    "reason": "changed",
                    "breakpoint": {
                        "id": bp.id,
                        "verified": bp.verified,
                        "line": bp.line,
                        "message": bp.message,
                    },
                },
            )
        return bound

    def set_function_breakpoints(self, session_id: str, names: list[str]) -> list[Breakpoint]:
        session = self._session(session_id)
        out: list[Breakpoint] = []
        with session.lock:
            for name in names:
                bp = Breakpoint(id=session.next_bp_id, source_uri=session.source_uri, line=1)
                session.next_bp_id += 1
                fn = next((f for f in session.module.functions if f.name == name), None)
                if fn is None or not fn.blocks or not fn.blocks[0].instructions:
                    bp.verified = False
                    bp.message = "function not found or not executable"
                    out.append(bp)
                    continue
                instr = fn.blocks[0].instructions[0]
                point = ExecPoint(kind="HLIR", function=fn.name, block=fn.blocks[0].label, op_id=instr.instr_id or "")
                bp.point = point
                bp.verified = True
                prov = session.source_map.bind(point)
                if prov and prov.primary_span:
                    bp.line = prov.primary_span.start_line
                session.breakpoints_by_point.setdefault(point.key(), []).append(bp)
                out.append(bp)
        return out

    def set_exception_breakpoints(self, session_id: str, filters: list[str]) -> None:
        session = self._session(session_id)
        with session.lock:
            session.exception_filters = set(filters or ["all"])

    def get_threads(self, session_id: str) -> list[dict[str, Any]]:
        self._session(session_id)
        return [{"id": 1, "name": "main"}]

    def get_stack(self, session_id: str, thread_id: int, start: int = 0, levels: int = 20) -> list[dict[str, Any]]:
        del thread_id
        session = self._session(session_id)
        with session.lock:
            frames = sorted(session.frame_tokens.values(), key=lambda f: f.depth, reverse=True)
            sliced = frames[start : start + levels]
            out: list[dict[str, Any]] = []
            for frame in sliced:
                span = frame.provenance.primary_span if frame.provenance else None
                out.append(
                    {
                        "id": frame.id,
                        "name": frame.function,
                        "line": span.start_line if span else 1,
                        "column": span.start_col if span else 1,
                        "endLine": span.end_line if span else 1,
                        "endColumn": span.end_col if span else 1,
                        "source": {"name": Path(session.source_uri).name, "path": session.source_uri},
                    }
                )
            return out

    def get_scopes(self, session_id: str, frame_id: int) -> list[dict[str, Any]]:
        session = self._session(session_id)
        with session.lock:
            frame = session.frame_tokens.get(frame_id)
            if frame is None:
                return []

            scopes: list[dict[str, Any]] = []
            for kind, payload in [
                ("Arguments", frame.args),
                ("Locals", frame.locals),
                ("Temporaries", frame.temporaries),
                ("Globals", session.globals),
            ]:
                sid = session.next_scope_id
                session.next_scope_id += 1
                session.scope_tokens[sid] = (kind, frame_id)
                ref = self._value_ref_locked(session, payload, kind.lower())
                scopes.append(
                    {
                        "name": kind,
                        "presentationHint": "locals" if kind in {"Locals", "Arguments"} else "registers",
                        "variablesReference": ref,
                        "expensive": False,
                    }
                )
            return scopes

    def get_variables(self, session_id: str, variables_ref: int, start: int = 0, count: int = 0) -> list[dict[str, Any]]:
        session = self._session(session_id)
        with session.lock:
            ref = session.value_tokens.get(variables_ref)
            if ref is None:
                return []

            value = ref.backing_object
            items: list[tuple[str, Any]]
            if isinstance(value, dict):
                items = [(str(k), v) for k, v in value.items()]
            elif isinstance(value, list):
                items = [(str(i), v) for i, v in enumerate(value)]
            else:
                return []

            if count > 0:
                items = items[start : start + count]
            elif start > 0:
                items = items[start:]

            out: list[dict[str, Any]] = []
            for name, item in items:
                out.append(self._format_variable_locked(session, name, item))
            return out

    def evaluate(self, session_id: str, expr: str, frame_id: int | None, context: str = "watch") -> dict[str, Any]:
        session = self._session(session_id)
        if context in {"watch", "hover", "repl"}:
            with session.lock:
                env = self._eval_env_for_frame_locked(session, frame_id)
            value = _safe_eval(expr, env)
            result = self._value_to_string(value)
            out_ref = 0
            with session.lock:
                out_ref = self._value_ref_locked(session, value, "eval") if isinstance(value, (list, dict)) else 0
            return {"result": result, "type": type(value).__name__, "variablesReference": out_ref}
        raise RuntimeError(f"unsupported evaluate context: {context}")

    def get_source(self, session_id: str, source_reference: int) -> str:
        session = self._session(session_id)
        if source_reference > 0:
            with session.lock:
                generated = session.generated_sources.get(source_reference)
                if generated is None:
                    raise RuntimeError(f"unknown sourceReference: {source_reference}")
                return generated["text"]
        return Path(session.source_uri).read_text(encoding="utf-8")

    def _run_session(self, session: _RuntimeSession) -> None:
        assert session.interpreter is not None
        try:
            with session.cond:
                session.state = "Running"

            result = session.interpreter.run_module(session.module, entry="main")

            if session.trace_compare:
                self._trace_compare_kernel(session)

            with session.cond:
                if session.state != "Terminated":
                    session.state = "Terminated"
            self._emit_event(session, "exited", {"exitCode": int(result.value) if isinstance(result.value, int) else 0})
            self._emit_event(session, "terminated", {})
        except ManvError as err:
            with session.cond:
                session.last_error = err.render()
                if "all" in session.exception_filters:
                    self._stop_locked(session, "exception", session.last_error, session.current_point, session.current_provenance)
                    while session.state == "Stopped" and not session.terminate_requested:
                        session.cond.wait()
                session.state = "Terminated"
            self._emit_event(session, "terminated", {})
        except Exception as err:
            with session.cond:
                session.last_error = str(err)
                if "all" in session.exception_filters:
                    self._stop_locked(session, "exception", session.last_error, session.current_point, session.current_provenance)
                    while session.state == "Stopped" and not session.terminate_requested:
                        session.cond.wait()
                session.state = "Terminated"
            self._emit_event(session, "terminated", {})

    def _trace_compare_kernel(self, session: _RuntimeSession) -> None:
        tracer = GraphCaptureTracer()
        out = io.StringIO()
        interp = HLIRInterpreter(stdout=out, tracer=tracer)
        interp.run_module(session.module, entry="main")
        graph = tracer.to_graph_ir()
        kernel = lower_graph_to_kernel(graph)
        exec_result = execute_kernel_ir(kernel, include_trace=True)

        traces = exec_result.get("trace", {})
        for kernel_desc in kernel.get("kernels", []):
            kernel_name = str(kernel_desc.get("kernel_name"))
            actual_trace = traces.get(kernel_name, {})
            for op in kernel_desc.get("ops", []):
                op_id = str(op.get("id"))
                attrs = op.get("attrs", {})
                expected = attrs.get("result")
                if expected is None:
                    continue
                actual = actual_trace.get(op_id)
                if expected != actual:
                    prov = provenance_from_kernel_op(op)
                    point = ExecPoint(kind="KIR", function=kernel_name, block="kernel", op_id=op_id)
                    message = (
                        f"Kernel mismatch at op {op_id}: expected={expected!r}, actual={actual!r}"
                    )
                    with session.cond:
                        self._stop_locked(session, "exception", message, point, prov)
                        while session.state == "Stopped" and not session.terminate_requested:
                            session.cond.wait()
                    return

    def _set_running_locked(self, session: _RuntimeSession) -> None:
        session.state = "Running"
        self._emit_event(session, "continued", {"threadId": 1, "allThreadsContinued": True})
        session.cond.notify_all()

    def _set_step_plan_locked(self, session: _RuntimeSession, mode: str) -> None:
        if session.current_point is None:
            session.step_plan = None
            return
        current_frame_depth = max(0, len(session.frame_tokens) - 1)
        line = 1
        if session.current_provenance and session.current_provenance.primary_span:
            line = session.current_provenance.primary_span.start_line
        session.step_plan = StepPlan(
            mode=mode,
            start_frame_depth=current_frame_depth,
            start_point_key=session.current_point.key(),
            start_line=line,
        )

    def _step_boundary_reached(self, session: _RuntimeSession, point: ExecPoint, prov: Provenance | None) -> bool:
        plan = session.step_plan
        if plan is None:
            return False

        current_depth = max(0, len(session.frame_tokens) - 1)
        current_line = prov.primary_span.start_line if prov and prov.primary_span else -1

        if point.key() == plan.start_point_key and current_depth == plan.start_frame_depth:
            return False

        if plan.mode == "In":
            return current_line != plan.start_line or current_depth != plan.start_frame_depth
        if plan.mode == "Over":
            return current_depth <= plan.start_frame_depth and current_line != plan.start_line
        if plan.mode == "Out":
            return current_depth < plan.start_frame_depth
        return False

    def _stop_locked(
        self,
        session: _RuntimeSession,
        reason: str,
        message: str,
        point: ExecPoint | None,
        provenance: Provenance | None,
    ) -> None:
        session.state = "Stopped"
        session.stop_reason = reason
        session.stop_message = message
        session.current_point = point
        session.current_provenance = provenance
        self._refresh_frames_locked(session)

        body: dict[str, Any] = {"reason": reason, "threadId": 1, "allThreadsStopped": True}
        if message:
            body["text"] = message
        self._emit_event(session, "stopped", body)

    def _refresh_frames_locked(self, session: _RuntimeSession) -> None:
        interp = session.interpreter
        if interp is None:
            return

        frames = list(interp.call_stack)
        session.frame_tokens.clear()
        session.scope_tokens.clear()
        session.value_tokens.clear()
        session.next_scope_id = 1
        session.next_value_id = 1

        function_params = {fn.name: [str(p.get("name")) for p in fn.params] for fn in session.module.functions}

        for depth, frame in enumerate(frames):
            frame_id = session.next_frame_id
            session.next_frame_id += 1

            params = function_params.get(frame.function, [])
            args_dict = {params[i] if i < len(params) else f"arg{i}": v for i, v in enumerate(frame.args)}

            point = ExecPoint(kind="HLIR", function=frame.function, block=frame.block, op_id=frame.current_instr_id or "")
            prov = session.source_map.bind(point)

            logical = LogicalFrame(
                id=frame_id,
                function=frame.function,
                block=frame.block,
                depth=depth,
                provenance=prov,
                args=args_dict,
                locals=dict(frame.vars_mem),
                temporaries=dict(frame.values),
            )
            session.frame_tokens[frame_id] = logical

    def _matching_breakpoint_locked(self, session: _RuntimeSession, point: ExecPoint) -> Breakpoint | None:
        if not point.op_id:
            return None
        bps = session.breakpoints_by_point.get(point.key(), [])
        if not bps:
            return None
        return bps[0]

    def _breakpoint_should_stop(self, bp: Breakpoint, env: dict[str, Any]) -> bool:
        bp.hits += 1

        if bp.hit_condition:
            cond = bp.hit_condition.strip()
            if cond.isdigit() and bp.hits < int(cond):
                return False
            if cond.startswith(">=") and cond[2:].strip().isdigit() and bp.hits < int(cond[2:].strip()):
                return False

        if bp.condition:
            try:
                value = _safe_eval(bp.condition, env)
            except Exception:
                return False
            return bool(value)
        return True

    def _current_eval_env_locked(self, session: _RuntimeSession) -> dict[str, Any]:
        frames = sorted(session.frame_tokens.values(), key=lambda f: f.depth)
        env: dict[str, Any] = dict(session.globals)
        if frames:
            top = frames[-1]
            env.update(top.args)
            env.update(top.locals)
            env.update(top.temporaries)
        env["len"] = len
        return env

    def _eval_env_for_frame_locked(self, session: _RuntimeSession, frame_id: int | None) -> dict[str, Any]:
        env: dict[str, Any] = dict(session.globals)
        if frame_id is None:
            env.update(self._current_eval_env_locked(session))
        else:
            frame = session.frame_tokens.get(frame_id)
            if frame is not None:
                env.update(frame.args)
                env.update(frame.locals)
                env.update(frame.temporaries)
        env["len"] = len
        return env

    def _format_variable_locked(self, session: _RuntimeSession, name: str, value: Any) -> dict[str, Any]:
        ref = self._value_ref_locked(session, value, "child") if isinstance(value, (list, dict, InstanceObject)) else 0
        return {
            "name": name,
            "value": self._value_to_string(value),
            "type": type(value).__name__,
            "variablesReference": ref,
        }

    def _value_ref_locked(self, session: _RuntimeSession, value: Any, kind: str) -> int:
        vid = session.next_value_id
        session.next_value_id += 1
        paging = None
        if isinstance(value, list):
            paging = {"length": len(value)}
        if isinstance(value, dict):
            paging = {"length": len(value)}
        if isinstance(value, InstanceObject):
            paging = {"length": len(value.attrs)}
        session.value_tokens[vid] = ValueRef(id=vid, value_kind=kind, backing_object=value, paging_info=paging)
        return vid

    def _value_to_string(self, value: Any) -> str:
        if isinstance(value, list):
            preview = ", ".join(repr(v) for v in value[:8])
            suffix = ", ..." if len(value) > 8 else ""
            return f"array(len={len(value)}) [{preview}{suffix}]"
        if isinstance(value, dict):
            keys = sorted(value.keys(), key=lambda k: repr(k))[:8]
            preview = ", ".join(f"{k!r}: {value[k]!r}" for k in keys)
            suffix = ", ..." if len(value) > 8 else ""
            return f"map(len={len(value)}) {{{preview}{suffix}}}"
        if isinstance(value, InstanceObject):
            keys = sorted(value.attrs.keys())[:8]
            preview = ", ".join(f"{k}={value.attrs[k]!r}" for k in keys)
            suffix = ", ..." if len(value.attrs) > 8 else ""
            return f"{value.type_obj.name}(attrs={len(value.attrs)}) {{{preview}{suffix}}}"
        if isinstance(value, ExceptionObject):
            return f"{value.type_obj.name}: {value.message}"
        return repr(value)

    def _emit_event(self, session: _RuntimeSession, event: str, body: dict[str, Any]) -> None:
        payload = {"event": event, "body": body, "sessionId": session.id}
        with session.lock:
            session.event_history.append(payload)
            listeners = list(session.listeners)
        for listener in listeners:
            try:
                listener(payload)
            except Exception:
                continue

    def _register_generated_source_locked(self, session: _RuntimeSession, *, name: str, text: str) -> int:
        for ref, existing in session.generated_sources.items():
            if existing.get("name") == name and existing.get("text") == text:
                return ref
        ref = session.next_source_ref
        session.next_source_ref += 1
        session.generated_sources[ref] = {"name": name, "text": text}
        return ref

    def _session(self, session_id: str) -> _RuntimeSession:
        if session_id not in self._sessions:
            raise RuntimeError(f"debug session not found: {session_id}")
        return self._sessions[session_id]


def _safe_eval(expr: str, env: dict[str, Any]) -> Any:
    translated = _translate_expr(expr)
    node = pyast.parse(translated, mode="eval")
    return _eval_node(node.body, env)


def _translate_expr(expr: str) -> str:
    out = expr.replace("&&", " and ").replace("||", " or ")
    # Replace standalone ! with not while preserving !=.
    out = out.replace("!=", "__NEQ__")
    out = out.replace("!", " not ")
    out = out.replace("__NEQ__", "!=")
    return out


def _eval_node(node: pyast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, pyast.Constant):
        return node.value
    if isinstance(node, pyast.Name):
        if node.id not in env:
            raise RuntimeError(f"undefined symbol: {node.id}")
        return env[node.id]
    if isinstance(node, pyast.List):
        return [_eval_node(elt, env) for elt in node.elts]
    if isinstance(node, pyast.Tuple):
        return tuple(_eval_node(elt, env) for elt in node.elts)
    if isinstance(node, pyast.Dict):
        return {_eval_node(k, env): _eval_node(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, pyast.UnaryOp):
        value = _eval_node(node.operand, env)
        if isinstance(node.op, pyast.USub):
            return -value
        if isinstance(node.op, pyast.Not):
            return not bool(value)
        raise RuntimeError("unsupported unary operator")
    if isinstance(node, pyast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        if isinstance(node.op, pyast.Add):
            return left + right
        if isinstance(node.op, pyast.Sub):
            return left - right
        if isinstance(node.op, pyast.Mult):
            return left * right
        if isinstance(node.op, pyast.Div):
            return left / right
        if isinstance(node.op, pyast.Mod):
            return left % right
        raise RuntimeError("unsupported binary operator")
    if isinstance(node, pyast.BoolOp):
        values = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, pyast.And):
            return all(bool(v) for v in values)
        if isinstance(node.op, pyast.Or):
            return any(bool(v) for v in values)
        raise RuntimeError("unsupported boolean operator")
    if isinstance(node, pyast.Compare):
        left = _eval_node(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, env)
            if isinstance(op, pyast.Eq):
                ok = left == right
            elif isinstance(op, pyast.NotEq):
                ok = left != right
            elif isinstance(op, pyast.Lt):
                ok = left < right
            elif isinstance(op, pyast.LtE):
                ok = left <= right
            elif isinstance(op, pyast.Gt):
                ok = left > right
            elif isinstance(op, pyast.GtE):
                ok = left >= right
            else:
                raise RuntimeError("unsupported comparison")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, pyast.Subscript):
        target = _eval_node(node.value, env)
        if isinstance(node.slice, pyast.Slice):
            lower = _eval_node(node.slice.lower, env) if node.slice.lower is not None else None
            upper = _eval_node(node.slice.upper, env) if node.slice.upper is not None else None
            return target[lower:upper]
        return target[_eval_node(node.slice, env)]
    if isinstance(node, pyast.Call):
        if not isinstance(node.func, pyast.Name) or node.func.id != "len":
            raise RuntimeError("function calls are not allowed in watch evaluation")
        args = [_eval_node(a, env) for a in node.args]
        return len(args[0])
    raise RuntimeError("unsupported expression")


