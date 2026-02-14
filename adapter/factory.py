# adapters/factory.py

from operate.models.apis import (
    get_next_action,  # function to get next action
    call_gpt_4o, 
    call_qwen_vl_with_ocr, 
    call_gpt_4o_with_ocr, 
    call_o1_with_ocr,
    call_claude_3_with_ocr,
    call_gemini_pro_vision,
    call_ollama_llava,
    call_gpt_4_1_with_ocr,
    call_gpt_4o_labeled,
)
from operate.exceptions import ModelNotRecognizedException


class AdapterFactory:
    """
    Adapter Factory for dynamic model selection.
    It fetches the model logic from apis.py based on the model name.
    """

    @staticmethod
    def create_llm_callable(model_name: str):
        """
        This method dynamically resolves the model logic from `apis.py`.
        
        Args:
            model_name (str): The name of the model to use.
        
        Returns:
            Callable: The function that implements the model's logic.
        
        Raises:
            ModelNotRecognizedException: If the model is not recognized.
        """
        if model_name == "gpt-4o":
            return call_gpt_4o
        elif model_name == "qwen-vl":
            return call_qwen_vl_with_ocr
        elif model_name == "gpt-4o-with-ocr":
            return call_gpt_4o_with_ocr
        elif model_name == "o1-with-ocr":
            return call_o1_with_ocr
        elif model_name == "claude-3":
            return call_claude_3_with_ocr
        elif model_name == "gemini-pro-vision":
            return call_gemini_pro_vision
        elif model_name == "llava":
            return call_ollama_llava
        elif model_name == "gpt-4_1-with-ocr":
            return call_gpt_4_1_with_ocr
        elif model_name == "gpt-4o-labeled":
            return call_gpt_4o_labeled
        else:
            raise ModelNotRecognizedException(f"Model '{model_name}' not recognized!")

    @staticmethod
    def get_action(model_name: str, messages, objective, session_id):
        """
        Fetches the next action for the given model.
        
        Args:
            model_name (str): The model name.
            messages (list): List of messages to pass to the model.
            objective (str): The objective for the model.
            session_id (str): Unique session identifier.
        
        Returns:
            Action or Exception: The action suggested by the model.
        """
        try:
            action_function = AdapterFactory.create_llm_callable(model_name)
            return get_next_action(action_function, messages, objective, session_id)
        except ModelNotRecognizedException as e:
            print(f"Error: {str(e)}")
            raise
