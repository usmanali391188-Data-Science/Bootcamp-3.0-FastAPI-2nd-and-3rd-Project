from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import uuid
import requests
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="Chatbot API (Gemini + Hugging Face)")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


USERS_FILE = "users.json"
HF_CHATS_FILE = "hf_chats.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "You_need_to_use_your_own_key_here")
genai.configure(api_key=GEMINI_API_KEY)

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "You_need_to_use_your_own_key_here")
HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"



def load_json(file_path, default):
    """Load JSON safely; create file if missing."""
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(default, f)
        return default
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, type(default)) else default
    except (json.JSONDecodeError, FileNotFoundError):
        with open(file_path, "w") as f:
            json.dump(default, f)
        return default


def save_json(file_path, data):
    """Save data to JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)



@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    users = load_json(USERS_FILE, [])
    if any(u["username"] == username for u in users):
        raise HTTPException(status_code=400, detail="Username already exists")

    token = str(uuid.uuid4())
    users.append({"username": username, "password": password, "token": token})
    save_json(USERS_FILE, users)
    return {"username": username, "token": token}


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    users = load_json(USERS_FILE, [])
    user = next((u for u in users if u["username"] == username and u["password"] == password), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"username": user["username"], "token": user["token"]}



@app.post("/chat-gemini")
def chat_gemini(prompt: str = Form(...), token: str = Form(...)):
    users = load_json(USERS_FILE, [])
    user = next((u for u in users if u["token"] == token), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt, request_options={"timeout": 60})

        if not response or not hasattr(response, "text"):
            raise HTTPException(status_code=500, detail="Empty response from Gemini")

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")



@app.post("/chat-huggingface")
def chat_huggingface(prompt: str = Form(...), token: str = Form(...)):
    users = load_json(USERS_FILE, [])
    user = next((u for u in users if u["token"] == token), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 128
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Hugging Face API error: {response.text}")

        data = response.json()

        if isinstance(data, dict) and "choices" in data:
            reply = data["choices"][0]["message"]["content"]
        elif isinstance(data, dict) and "generated_text" in data:
            reply = data["generated_text"]
        else:
            reply = str(data)

        hf_chats = load_json(HF_CHATS_FILE, [])
        hf_chats.append({
            "user": user["username"],
            "prompt": prompt,
            "response": reply
        })
        save_json(HF_CHATS_FILE, hf_chats)

        return {"response": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face API error: {e}")



class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 500


@app.post("/generate")
def generate_text(request: PromptRequest):
    MODEL_ID = HF_MODEL
    ENDPOINT = HF_API_URL

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Hugging Face API: {e}")

    data = response.json()
    try:
        generated_text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected response format: {data}")

    return {"generated_text": generated_text}
