from flask import Flask, render_template, request, jsonify
from chatbot.model import ChatbotModel
from chatbot.conversation import ConversationManager

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load datasets (for exploration or preprocessing, not for direct inference)
owt = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
c4 = load_dataset("c4", "en", split="train", streaming=True, trust_remote_code=True)
wiki = load_dataset("wiki40b", "en")
books = load_dataset("bookcorpus", trust_remote_code=True)
code = load_dataset("codeparrot/codeparrot-clean")
gsm8k = load_dataset("gsm8k", "main", split="train")
csqa = load_dataset("commonsense_qa")
piqa = load_dataset("piqa", trust_remote_code=True)


app = Flask(__name__)
chatbot_model = ChatbotModel()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = chatbot_model.generate_response(user_input)
    return jsonify({"response": response})


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

     model_name = "deepseek-ai/deepseek-llm-14b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    print("Welcome to DeepSeek 14B Chatbot! Type 'quit' to exit.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        chat_history.append({"role": "user", "content": user_input})

        # Build prompt from chat history
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
            max_length=inputs["input_ids"].shape[1] + 128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new AI response
        ai_reply = response[len(prompt):].split("User:")[0].strip()
        print("AI:", ai_reply)
        chat_history.append({"role": "ai", "content": ai_reply})


if __name__ == "__main__":
    app.run(debug=True, port=7854)
    main()
