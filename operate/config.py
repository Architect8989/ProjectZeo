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
    Examples:
      - openai/gpt-4o-mini
      - anthropic/claude-3.5-sonnet
      - qwen/qwen2.5-vl-72b-instruct
    """
    return "/" in model


class Config:
    """
    Configuration class for managing settings.

    Attributes:
        verbose (bool): Flag indicating whether verbose mode is enabled.
        openai_api_key (str): API key for OpenAI.
        google_api_key (str): API key for Google.
        ollama_host (str): url to ollama running remotely.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
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
        if self.verbose:
            print("[Config][initialize_openai]")

        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")

        client = OpenAI(api_key=api_key)
        client.api_key = api_key
        client.base_url = os.getenv("OPENAI_API_BASE_URL", client.base_url)
        return client

    def initialize_qwen(self):
        if self.verbose:
            print("[Config][initialize_qwen]")

        api_key = self.qwen_api_key or os.getenv("QWEN_API_KEY")

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client.api_key = api_key
        client.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return client

    def initialize_google(self):
        api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key, transport="rest")
        return genai.GenerativeModel("gemini-pro-vision")

    def initialize_ollama(self):
        self.ollama_host = self.ollama_host or os.getenv("OLLAMA_HOST", None)
        return Client(host=self.ollama_host)

    def initialize_anthropic(self):
        api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)

    # -------------------------
    # VALIDATION (EXTENDED, NOT REPLACED)
    # -------------------------

    def validation(self, model, voice_mode):
        """
        Validate the input parameters for the dialog operation.

        Existing SOC logic is preserved.
        OpenRouter models bypass provider-specific API-key enforcement.
        """

        # ---- NEW: OpenRouter bypass (ADDITIVE ONLY) ----
        if is_openrouter_model(model):
            if self.verbose:
                print(
                    "[Config][validation] OpenRouter model detected. "
                    "Skipping provider-specific API key checks."
                )
            return

        # ---- LEGACY SOC VALIDATION (UNCHANGED) ----
        self.require_api_key(
            "OPENAI_API_KEY",
            "OpenAI API key",
            model == "gpt-4"
            or voice_mode
            or model == "gpt-4-with-som"
            or model == "gpt-4-with-ocr"
            or model == "gpt-4.1-with-ocr"
            or model == "o1-with-ocr",
        )

        self.require_api_key(
            "GOOGLE_API_KEY",
            "Google API key",
            model == "gemini-pro-vision",
        )

        self.require_api_key(
            "ANTHROPIC_API_KEY",
            "Anthropic API key",
            model == "claude-3",
        )

        self.require_api_key(
            "QWEN_API_KEY",
            "Qwen API key",
            model == "qwen-vl",
        )

    # -------------------------
    # API KEY HANDLING (UNCHANGED)
    # -------------------------

    def require_api_key(self, key_name, key_description, is_required):
        key_exists = bool(os.environ.get(key_name))
        if self.verbose:
            print("[Config] require_api_key")
            print("[Config] key_name", key_name)
            print("[Config] key_description", key_description)
            print("[Config] key_exists", key_exists)

        if is_required and not key_exists:
            self.prompt_and_save_api_key(key_name, key_description)

    def prompt_and_save_api_key(self, key_name, key_description):
        key_value = input_dialog(
            title="API Key Required",
            text=f"Please enter your {key_description}:",
        ).run()

        if key_value is None:
            sys.exit("Operation cancelled by user.")

        if key_value:
            if key_name == "OPENAI_API_KEY":
                self.openai_api_key = key_value
            elif key_name == "GOOGLE_API_KEY":
                self.google_api_key = key_value
            elif key_name == "ANTHROPIC_API_KEY":
                self.anthropic_api_key = key_value
            elif key_name == "QWEN_API_KEY":
                self.qwen_api_key = key_value

            self.save_api_key_to_env(key_name, key_value)
            load_dotenv()

    @staticmethod
    def save_api_key_to_env(key_name, key_value):
        with open(".env", "a") as file:
            file.write(f"\n{key_name}='{key_value}'")
