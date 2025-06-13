from pathlib import Path

class Config:
    MODEL_NAME = "your-advanced-chatbot-model"
    MAX_LENGTH = 100
    TEMPERATURE = 0.7
    TOP_K = 50
    TOP_P = 0.95
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    API_KEY = "your_api_key_here"  # If applicable
    LOGGING_ENABLED = True
    LOG_FILE_PATH = Path("logs/conversation.log")  # Path for logging conversation data

    @staticmethod
    def get_model_params():
        return {
            "max_length": Config.MAX_LENGTH,
            "temperature": Config.TEMPERATURE,
            "top_k": Config.TOP_K,
            "top_p": Config.TOP_P,
        }