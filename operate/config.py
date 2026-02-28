import os
import sys
import threading

from dotenv import load_dotenv

# FIX: ollama import is deferred to initialize_ollama() to avoid ImportError
# on systems where ollama is not yet installed (e.g. fresh setup, CI).
# The Client is only needed when actually calling initialize_ollama().


def is_openrouter_model(model: str) -> bool:
    """
    Detect OpenRouter-style models (contain '/', but not the 'ollama/' prefix).
    e.g. 'openai/gpt-4o' is OpenRouter; 'ollama/qwen2.5-vl' is local Ollama.
    """
    return "/" in model and not model.startswith("ollama/")


# PATCH Risk #6a: module-level lock protecting all .env writes.
_ENV_FILE_LOCK = threading.Lock()


class Config:
    """
    Process-global configuration singleton.

    All cloud SDK imports are deferred to the first call of their
    respective initialize_*() methods. This makes Config safe to import
    on any machine regardless of which optional cloud SDKs are installed.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        load_dotenv()

        self.verbose = False

        # API keys — all None until set by env, .env file, or interactive prompt.
        self.openai_api_key = None
        self.google_api_key = None
        self.ollama_host = None
        self.anthropic_api_key = None
        self.qwen_api_key = None

    # =========================================================
    # PROVIDER INITIALIZERS — all cloud SDKs lazy-imported
    # =========================================================

    def initialize_openai(self):
        """OpenAI client. Lazy-imports openai SDK."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK not installed. Run: pip install openai"
            ) from exc

        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        client.api_key = api_key
        client.base_url = os.getenv("OPENAI_API_BASE_URL", client.base_url)
        return client

    def initialize_qwen(self):
        """Qwen cloud client (DashScope). Reuses OpenAI-compatible SDK."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK not installed. Run: pip install openai"
            ) from exc

        api_key = self.qwen_api_key or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("QWEN_API_KEY not set")

        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def initialize_google(self):
        """Google Gemini client. Lazy-imports google-generativeai SDK."""
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai SDK not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key, transport="rest")
        return genai.GenerativeModel("gemini-pro-vision")

    def initialize_ollama(self):
        """Ollama client. Lazy-imports ollama SDK."""
        try:
            from ollama import Client  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "ollama SDK not installed. Run: pip install 'ollama>=0.3.0'"
            ) from exc
        self.ollama_host = self.ollama_host or os.getenv("OLLAMA_HOST")
        return Client(host=self.ollama_host)

    def initialize_anthropic(self):
        """Anthropic client. Lazy-imports anthropic SDK."""
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc

        api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        return anthropic.Anthropic(api_key=api_key)

    # =========================================================
    # VALIDATION — pure, no SDK calls, no side effects
    # =========================================================

    def validation(self, model: str, voice_mode: bool = False) -> None:
        """
        Check that required API keys are present for the chosen model.
        Does NOT initialise any SDK — purely reads env vars.
        """
        if not model:
            raise ValueError("Model must be specified")

        if is_openrouter_model(model):
            return

        self.require_api_key(
            "OPENAI_API_KEY",
            model in {
                "gpt-4", "gpt-4o", "gpt-4-with-som",
                "gpt-4-with-ocr", "gpt-4o-with-ocr",
                "gpt-4.1-with-ocr", "gpt-4o-labeled", "o1-with-ocr",
            } or voice_mode,
        )

        self.require_api_key(
            "GOOGLE_API_KEY",
            model == "gemini-pro-vision",
        )

        self.require_api_key(
            "ANTHROPIC_API_KEY",
            model in {"claude-3", "claude-3-opus", "claude-3-sonnet"},
        )

        self.require_api_key(
            "QWEN_API_KEY",
            model == "qwen-vl",
        )
        # Ollama-local models (qwen2.5-vl etc.) require no API key.

    # =========================================================
    # API KEY HANDLING
    # =========================================================

    def require_api_key(self, key_name: str, is_required: bool) -> None:
        if is_required and not os.environ.get(key_name):
            self.prompt_and_save_api_key(key_name)

    def prompt_and_save_api_key(self, key_name: str) -> None:
        try:
            from prompt_toolkit.shortcuts import input_dialog  # noqa: PLC0415
            key_value = input_dialog(
                title="API Key Required",
                text=f"Please enter {key_name}:",
            ).run()
        except Exception:
            key_value = input(f"Enter {key_name}: ").strip()

        if not key_value:
            sys.exit("Operation cancelled by user.")

        setattr(self, key_name.lower(), key_value)
        self.save_api_key_to_env(key_name, key_value)
        load_dotenv()

    @staticmethod
    def save_api_key_to_env(key_name: str, key_value: str) -> None:
        """
        Idempotent .env write under lock (PATCH Risk #6a).
        Values are double-quoted with embedded quotes escaped (PATCH Risk #6b).
        """
        escaped_value = key_value.replace("\\", "\\\\").replace('"', '\\"')
        new_line = f'{key_name}="{escaped_value}"\n'

        with _ENV_FILE_LOCK:
            lines: list = []
            if os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    lines = [
                        ln for ln in f.readlines()
                        if not ln.startswith(f"{key_name}=")
                    ]
            lines.append(new_line)

            tmp_path = ".env.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, ".env")
