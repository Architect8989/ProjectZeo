import os
import sys

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


class Config:
    """
    Process-global configuration singleton.

    NOT thread-safe by design.
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
            model in {
                "gpt-4",
                "gpt-4-with-som",
                "gpt-4-with-ocr",
                "gpt-4.1-with-ocr",
                "o1-with-ocr",
            } or voice_mode,
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
    def save_api_key_to_env(key_name, key_value):
        """
        Idempotent .env write.
        Overwrites existing key if present.
        """
        lines = []
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                lines = [
                    l for l in f.readlines()
                    if not l.startswith(f"{key_name}=")
                ]

        lines.append(f"{key_name}='{key_value}'\n")

        with open(".env", "w") as f:
            f.writelines(lines)
