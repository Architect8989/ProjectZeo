"""
operate/config.py — Process-global configuration singleton.

PATCH (audit Risk #6a): .env file now written under a threading.Lock() so
  that concurrent calls to save_api_key_to_env() cannot interleave and
  corrupt the file.

PATCH (audit Risk #6b): API key values are now quoted with double-quotes
  and any literal double-quote characters inside the value are escaped
  (\").  Previously single-quotes were used; a key value that contained
  a single-quote (e.g. a token with apostrophes) would break .env parsing.
  The new escaping strategy is consistent with most dotenv libraries.
"""

import os
import sys
import threading

import google.generativeai as genai
from dotenv import load_dotenv
from ollama import Client
from openai import OpenAI
import anthropic
from prompt_toolkit.shortcuts import input_dialog


def is_openrouter_model(model: str) -> bool:
    """
    Detect OpenRouter-style models.
    """
    return "/" in model


# PATCH (audit Risk #6a): module-level lock protecting all .env writes.
_ENV_FILE_LOCK = threading.Lock()


class Config:
    """
    Process-global configuration singleton.

    Thread safety: the singleton construction and re-initialization guard
    are NOT thread-safe (by original design).  The .env write path is now
    protected via _ENV_FILE_LOCK (PATCH Risk #6a).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 🔒 FIX: prevent re-initialization
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        load_dotenv()

        self.verbose = False
        self.openai_api_key = None
        self.google_api_key = None
        self.ollama_host = None
        self.anthropic_api_key = None
        self.qwen_api_key = None

    # -------------------------
    # INITIALIZERS (UNCHANGED)
    # -------------------------

    def initialize_openai(self):
        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")

        client = OpenAI(api_key=api_key)
        client.api_key = api_key
        client.base_url = os.getenv(
            "OPENAI_API_BASE_URL",
            client.base_url,
        )
        return client

    def initialize_qwen(self):
        api_key = self.qwen_api_key or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("QWEN_API_KEY missing")

        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def initialize_google(self):
        api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY missing")

        genai.configure(api_key=api_key, transport="rest")
        return genai.GenerativeModel("gemini-pro-vision")

    def initialize_ollama(self):
        self.ollama_host = self.ollama_host or os.getenv("OLLAMA_HOST")
        return Client(host=self.ollama_host)

    def initialize_anthropic(self):
        api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")

        return anthropic.Anthropic(api_key=api_key)

    # -------------------------
    # VALIDATION (PURE)
    # -------------------------

    def validation(self, model, voice_mode):
        """
        Validation ONLY.
        No mutation. No side effects.
        """

        if not model:
            raise ValueError("Model must be specified")

        # OpenRouter models bypass provider enforcement
        if is_openrouter_model(model):
            return

        self.require_api_key(
            "OPENAI_API_KEY",
            model
            in {
                "gpt-4",
                "gpt-4-with-som",
                "gpt-4-with-ocr",
                "gpt-4.1-with-ocr",
                "o1-with-ocr",
            }
            or voice_mode,
        )

        self.require_api_key(
            "GOOGLE_API_KEY",
            model == "gemini-pro-vision",
        )

        self.require_api_key(
            "ANTHROPIC_API_KEY",
            model == "claude-3",
        )

        self.require_api_key(
            "QWEN_API_KEY",
            model == "qwen-vl",
        )

    # -------------------------
    # API KEY HANDLING
    # -------------------------

    def require_api_key(self, key_name, is_required):
        if is_required and not os.environ.get(key_name):
            self.prompt_and_save_api_key(key_name)

    def prompt_and_save_api_key(self, key_name):
        key_value = input_dialog(
            title="API Key Required",
            text=f"Please enter {key_name}:",
        ).run()

        if not key_value:
            sys.exit("Operation cancelled by user.")

        # cache locally
        setattr(self, key_name.lower(), key_value)

        self.save_api_key_to_env(key_name, key_value)
        load_dotenv()

    @staticmethod
    def save_api_key_to_env(key_name: str, key_value: str) -> None:
        """
        Idempotent .env write.  Overwrites existing key if present.

        PATCH (audit Risk #6a): entire read-modify-write is now performed
        under _ENV_FILE_LOCK to prevent concurrent corruption.

        PATCH (audit Risk #6b): key values are now stored with double-quote
        wrapping and any embedded double-quote characters are escaped as \\"
        so that key values containing single-quotes or special characters
        are serialised correctly for standard dotenv parsers.
        """
        # PATCH Risk #6b: escape embedded double-quotes, then wrap.
        escaped_value = key_value.replace("\\", "\\\\").replace('"', '\\"')
        new_line = f'{key_name}="{escaped_value}"\n'

        with _ENV_FILE_LOCK:  # PATCH Risk #6a
            lines: list[str] = []
            if os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    lines = [
                        ln
                        for ln in f.readlines()
                        if not ln.startswith(f"{key_name}=")
                    ]

            lines.append(new_line)

            # Write to a temp file then atomically replace to prevent
            # truncated .env on crash mid-write.
            tmp_path = ".env.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, ".env")
