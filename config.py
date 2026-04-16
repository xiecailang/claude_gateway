import os

# Resolve paths relative to this file's directory (the gateway root)
_gateway_dir = os.path.dirname(os.path.abspath(__file__))
_task_id_file = os.path.join(_gateway_dir, ".current_task_id")


def _resolve_log_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_gateway_dir, path)


def _get_task_id() -> str | None:
    """Read task_id from file at call time."""
    if os.path.exists(_task_id_file):
        try:
            with open(_task_id_file) as f:
                tid = f.read().strip()
                return tid or None
        except Exception:
            pass
    return None


class GatewayConfig:
    upstream_base_url: str = os.getenv("VLLM_UPSTREAM_URL", "http://127.0.0.1:8000")
    upstream_model: str = os.getenv(
        "VLLM_UPSTREAM_MODEL",
        "/home/nvidia/my_project/gpt-oss-120b/openai-mirror/gpt-oss-120b",
    )
    gateway_model: str = os.getenv("GATEWAY_MODEL", "vllm-local")
    port: int = int(os.getenv("GATEWAY_PORT", "9000"))
    timeout: float = float(os.getenv("GATEWAY_TIMEOUT", "300"))
    max_log_text_length: int = 100_000  # truncate logged text to 100KB

    @property
    def task_id(self) -> str | None:
        return _get_task_id()

    @property
    def log_file(self) -> str:
        tid = self.task_id
        if tid:
            return _resolve_log_path(f"logs/{tid}/requests.log")
        return _resolve_log_path(os.getenv("GATEWAY_LOG_FILE", "logs/requests.log"))

    @property
    def debug_log_file(self) -> str:
        tid = self.task_id
        if tid:
            return _resolve_log_path(f"logs/{tid}/gateway.log")
        base = os.getenv("GATEWAY_LOG_FILE", "logs/requests.log")
        return _resolve_log_path(os.path.join(os.path.dirname(base), "gateway.log"))

    @property
    def org_log_file(self) -> str:
        tid = self.task_id
        if tid:
            return _resolve_log_path(f"logs/{tid}/org_gateway.log")
        base = os.getenv("GATEWAY_LOG_FILE", "logs/requests.log")
        return _resolve_log_path(os.path.join(os.path.dirname(base), "org_gateway.log"))
