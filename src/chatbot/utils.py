from typing import List, Dict
import logging

def preprocess_input(user_input: str) -> str:
    # Basic preprocessing: strip whitespace and convert to lowercase
    return user_input.strip().lower()

def format_response(response: str) -> str:
    # Format the response for display
    return response.strip()

def log_conversation(user_input: str, bot_response: str, log_file: str = 'conversation.log') -> None:
    # Log the conversation to a file
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f'User: {user_input}')
    logging.info(f'Bot: {bot_response}')

def save_chat_history(chat_history: List[Dict[str, str]], file_path: str) -> None:
    # Save chat history to a file
    with open(file_path, 'w') as f:
        for entry in chat_history:
            f.write(f"User: {entry['user']}\n")
            f.write(f"Bot: {entry['bot']}\n")

def load_chat_history(file_path: str) -> List[Dict[str, str]]:
    # Load chat history from a file
    chat_history = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                user_line = lines[i].strip()
                bot_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
                chat_history.append({'user': user_line, 'bot': bot_line})
    except FileNotFoundError:
        pass  # If the file does not exist, return an empty history
    return chat_history