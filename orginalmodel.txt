import sys
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load datasets (for exploration or preprocessing, not for direct inference)
#owt = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
#c4 = load_dataset("c4", "en", split="train", streaming=True, trust_remote_code=True)
#wiki = load_dataset("wiki40b", "en")
#books = load_dataset("bookcorpus", trust_remote_code=True)
#code = load_dataset("codeparrot/codeparrot-clean")
#gsm8k = load_dataset("gsm8k", "main", split="train")
#csqa = load_dataset("commonsense_qa")
#piqa = load_dataset("piqa", trust_remote_code=True)

app = Flask(__name__)

# Load DeepSeek model and tokenizer once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "deepseek-ai/deepseek-llm-7b-chat"  # <-- Use a valid model name
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

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
        max_length=inputs["input_ids"].shape[1] + 500,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_reply = response[len(prompt):].split("User:")[0].strip()
    chat_history.append({"role": "ai", "content": ai_reply})
    return ai_reply

# Web chat history (per session, for demo use only)
chat_history_web = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = generate_deepseek_response(chat_history_web, user_input)
    return jsonify({"response": response})

def main():
    print("Welcome to DeepSeek 14B Chatbot! Type 'quit' to exit.")
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        ai_reply = generate_deepseek_response(chat_history, user_input)
        print("AI:", ai_reply)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        main()
    else:
        app.run(debug=True, port=7854)



if __name__ == "__main__":
    app.run(debug=True, port=7854)
    main()
