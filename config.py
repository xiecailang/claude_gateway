import os


class GatewayConfig:
    upstream_base_url: str = os.getenv("VLLM_UPSTREAM_URL", "http://127.0.0.1:8000")
    upstream_model: str = os.getenv(
        "VLLM_UPSTREAM_MODEL",
        "/home/nvidia/my_project/gpt-oss-120b/openai-mirror/gpt-oss-120b",
    )
    gateway_model: str = os.getenv("GATEWAY_MODEL", "vllm-local")
    port: int = int(os.getenv("GATEWAY_PORT", "9000"))
    log_file: str = os.getenv("GATEWAY_LOG_FILE", "logs/requests.log")
    timeout: float = float(os.getenv("GATEWAY_TIMEOUT", "300"))
    max_log_text_length: int = 100_000  # truncate logged text to 100KB
