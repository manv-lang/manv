from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import tomllib
from typing import Any
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from .diagnostics import ManvError, diag


DEFAULT_REGISTRY_URL = "https://registry.manv.dev"
AUTH_FILE_ENV = "MANV_AUTH_FILE"


@dataclass(frozen=True)
class RegistryAuth:
    registry: str
    token: str
    saved_at: str


def auth_file_path() -> Path:
    from os import environ

    override = environ.get(AUTH_FILE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".manv" / "auth.json").resolve()


def load_registry_auth() -> RegistryAuth | None:
    path = auth_file_path()
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    token = str(data.get("token", "")).strip()
    registry = str(data.get("registry", DEFAULT_REGISTRY_URL)).strip()
    if not token:
        return None

    return RegistryAuth(
        registry=_normalize_registry_url(registry or DEFAULT_REGISTRY_URL),
        token=token,
        saved_at=str(data.get("saved_at", "")),
    )


def save_registry_auth(*, registry: str, token: str) -> RegistryAuth:
    path = auth_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    session = RegistryAuth(
        registry=_normalize_registry_url(registry or DEFAULT_REGISTRY_URL),
        token=token.strip(),
        saved_at=datetime.now(UTC).isoformat(),
    )

    path.write_text(
        json.dumps(
            {
                "registry": session.registry,
                "token": session.token,
                "saved_at": session.saved_at,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return session


def clear_registry_auth() -> bool:
    path = auth_file_path()
    if not path.exists():
        return False
    path.unlink()
    return True


def ensure_manifest(path: Path, *, project_name: str, entry_rel: str) -> None:
    if path.exists():
        return

    template = (
        "[package]\n"
        f"name = \"{project_name}\"\n"
        "version = \"0.1.0\"\n"
        f"entry = \"{entry_rel}\"\n\n"
        "[build]\n"
        "target_dir = \".manv/target\"\n"
        "dist_dir = \"dist\"\n"
    )
    path.write_text(template, encoding="utf-8")


def add_dependency_entry(
    manifest_path: Path,
    *,
    dependency_name: str,
    payload: str | dict[str, Any],
) -> None:
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    deps = data.get("dependencies")
    if deps is None:
        deps = {}
    if not isinstance(deps, dict):
        raise ManvError(diag("E8101", "[dependencies] must be a table", str(manifest_path), 1, 1))

    deps[dependency_name] = payload
    data["dependencies"] = deps
    manifest_path.write_text(_dump_toml(data), encoding="utf-8")


def parse_registry_spec(spec: str) -> tuple[str, str | None]:
    raw = spec.strip()
    if not raw:
        raise ValueError("empty dependency spec")

    if "@" not in raw:
        return raw, None

    name, version = raw.rsplit("@", 1)
    name = name.strip()
    version = version.strip()
    if not name:
        raise ValueError("missing dependency name before '@'")
    if not version:
        return name, None
    return name, version


def looks_like_git_spec(spec: str) -> bool:
    s = spec.strip()
    return s.startswith("git@") or s.startswith("ssh://") or s.endswith(".git") or s.startswith("http://") or s.startswith("https://")


def infer_name_from_git_url(url: str) -> str:
    candidate = url.strip().rstrip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    name = candidate.rsplit("/", 1)[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    if not name:
        raise ValueError("could not infer dependency name from git URL")
    return name


def choose_registry_url(explicit_registry: str | None) -> str:
    if explicit_registry:
        return _normalize_registry_url(explicit_registry)

    auth = load_registry_auth()
    if auth:
        return auth.registry

    return DEFAULT_REGISTRY_URL


def choose_registry_token() -> str | None:
    auth = load_registry_auth()
    return auth.token if auth else None


def resolve_registry_version(name: str, *, requested: str | None, registry_url: str, token: str | None) -> str:
    if requested:
        return requested

    metadata = fetch_registry_package(name, registry_url=registry_url, token=token)
    if metadata is None:
        return "latest"

    for key in ("latest_version", "version"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    versions = metadata.get("versions")
    if isinstance(versions, list) and versions:
        first = versions[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
        if isinstance(first, dict):
            v = first.get("version")
            if isinstance(v, str) and v.strip():
                return v.strip()

    return "latest"


def fetch_registry_package(name: str, *, registry_url: str, token: str | None) -> dict[str, Any] | None:
    base = _normalize_registry_url(registry_url)
    safe_name = url_parse.quote(name)
    candidates = [
        f"{base}/api/v1/packages/{safe_name}",
        f"{base}/v1/packages/{safe_name}",
        f"{base}/packages/{safe_name}",
    ]

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for url in candidates:
        req = url_request.Request(url, headers=headers, method="GET")
        try:
            with url_request.urlopen(req, timeout=2.5) as response:
                text = response.read().decode("utf-8", errors="replace")
                payload = json.loads(text)
                if isinstance(payload, dict):
                    return payload
        except url_error.HTTPError as err:
            if err.code in {401, 403, 404}:
                continue
            continue
        except Exception:
            continue

    return None


def _normalize_registry_url(url: str) -> str:
    return url.strip().rstrip("/")


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    top_keys = list(data.keys())
    preferred = ["package", "build", "registries", "dependencies"]
    ordered_top = [k for k in preferred if k in top_keys] + [k for k in top_keys if k not in preferred]

    for key in ordered_top:
        value = data[key]
        if not isinstance(value, dict):
            lines.append(f"{_fmt_key(key)} = {_fmt_value(value)}")
    if lines:
        lines.append("")

    for key in ordered_top:
        value = data[key]
        if isinstance(value, dict):
            _emit_table(lines, [key], value)

    rendered = "\n".join(lines).strip() + "\n"
    return rendered


def _emit_table(lines: list[str], path: list[str], table: dict[str, Any]) -> None:
    header = ".".join(_fmt_key(k) for k in path)
    lines.append(f"[{header}]")

    keys = list(table.keys())
    for key in keys:
        value = table[key]
        if not isinstance(value, dict):
            lines.append(f"{_fmt_key(key)} = {_fmt_value(value)}")

    lines.append("")

    for key in keys:
        value = table[key]
        if isinstance(value, dict):
            _emit_table(lines, [*path, key], value)


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return f'"{value.replace("\\", "\\\\").replace("\"", "\\\"")}"'
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_value(v) for v in value) + "]"
    raise ValueError(f"unsupported TOML value type: {type(value).__name__}")


def _fmt_key(key: str) -> str:
    if re.match(r"^[A-Za-z0-9_-]+$", key):
        return key
    return f'"{key.replace("\\", "\\\\").replace("\"", "\\\"")}"'
