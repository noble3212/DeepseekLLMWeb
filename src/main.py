import sys
import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment or default path
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\user3\DeepSeek-R1-Distill-Qwen-14B") #please edit after you aquired a model

# Check if model path exists
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model path does not exist: {MODEL_PATH}")
    sys.exit(1)

# Set up Flask app
app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load tokenizer and model
print(f"[INFO] Loading model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
print("[INFO] Model loaded successfully.")

# Web session chat history
chat_history_web = []

# Core response generation function
def generate_deepseek_response(chat_history, user_input):
    chat_history.append({"role": "user", "content": user_input})
    prompt = ""
    for turn in chat_history:
        if turn["role"] == "user":
            prompt += f"User: {turn['content']}\n"
        else:
            prompt += f"AI: {turn['content']}\n"
    prompt += "AI:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_reply = response[len(prompt):].split("User:")[0].strip()
    chat_history.append({"role": "ai", "content": ai_reply})
    return ai_reply

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = generate_deepseek_response(chat_history_web, user_input)
    return jsonify({"response": response})

# CLI mode (optional)
def main_cli():
    print("Welcome to DeepSeek 14B Chatbot! Type 'quit' to exit.")
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        ai_reply = generate_deepseek_response(chat_history, user_input)
        print("AI:", ai_reply)

# Run app or CLI
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        main_cli()
    else:
        app.run(debug=True, port=7854)
