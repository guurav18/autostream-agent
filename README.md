# 🎬 AutoStream AI Sales Agent

A conversational AI agent built for **AutoStream** — an automated video editing SaaS platform for content creators. This agent identifies user intent, answers product questions using RAG, and captures high-intent leads automatically.

Built as part of the **ServiceHive / Inflx** Machine Learning Intern assignment.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- A Google Generative AI API key ([get one here](https://aistudio.google.com/app/apikey))

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set Your API Key
```bash
# Linux / macOS
export GOOGLE_API_KEY=your_api_key_here

# Windows (Command Prompt)
set GOOGLE_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your_api_key_here"
```

### Step 4 — Run the Agent
```bash
python agent.py
```

### Example Conversation
```
=======================================================
  AutoStream AI Sales Agent
  Powered by LangGraph + Gemini 2.5 Flash
=======================================================
  Type 'quit' to exit.

Agent: Hi! Welcome to AutoStream — AI-powered video editing for content creators. How can I help you today?

You: I want to sign up

Agent: That's great! To help you get started, could you please tell me your full name?

You: Gaurav

Agent: Thanks, Gaurav! What's your email address?

You: gaurav@example.com

Agent: Thanks, Gaurav! And what creator platform do you primarily use (e.g., YouTube, Instagram, TikTok)?

You: Instagram

==================================================
LEAD CAPTURED SUCCESSFULLY
==================================================
  Name     : Gaurav
  Email    : gaurav@example.com
  Platform : Instagram
==================================================

Agent: Excellent! Your details have been saved, Gaurav. Our team will be in touch soon to help you get started with AutoStream!
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why Gemini 2.5 Flash + LangGraph State Machine?

I chose **Gemini 2.5 Flash** because it offers excellent speed, cost-efficiency, and reasoning capability — ideal for real-time conversational agents. 

The agent uses **LangGraph's StateGraph** to implement an explicit state machine with typed dictionaries. This approach mirrors LangGraph's core concept — explicit nodes, conditional edges, and persistent state — backed by MemorySaver for conversation memory. Each "node" is a discrete function: intent classification, lead extraction, response generation, and tool execution.

### How State is Managed

State is maintained in a single `AgentState` TypedDict that persists across all conversation turns. It tracks:
- **Full message history** — passed to every API call so the model has complete context across 5–6+ turns
- **Lead collection fields** — `lead_name`, `lead_email`, `lead_platform` populated incrementally
- **Flags** — `collecting_lead` and `lead_captured` control which nodes activate

### Flow of Execution

```
User Input
    │
    ▼
Intent Classifier (Gemini 2.5 Flash)
    │
    ├─── casual_greeting  ──► Generate Response
    │
    ├─── product_inquiry  ──► RAG Lookup → Generate Response
    │
    └─── high_intent_lead ──► Lead Collector
                                    │
                                    ├─ (fields missing) ──► Ask Next Field
                                    │
                                    └─ (all collected)  ──► mock_lead_capture() ──► Done
```

### RAG Implementation

The knowledge base is stored in `knowledge_base.json` and loaded at startup. It is injected directly into the system prompt of every response-generation call — a simple but effective RAG pattern suitable for a structured, small knowledge base. For larger corpora, this would be replaced with vector embeddings + semantic search (e.g., ChromaDB + sentence-transformers).

---

## 📱 WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, we use the **WhatsApp Business API** (via Meta) with **webhooks** to receive and send messages in real-time.

### Architecture

```
WhatsApp User
     │
     ▼
WhatsApp Business API (Meta)
     │  (HTTP POST webhook)
     ▼
Your Backend Server  ◄──────────────────────────────────────┐
  (FastAPI / Flask)                                          │
     │                                                       │
     ├── Extract message + phone number                      │
     ├── Load/retrieve AgentState from Redis (by phone no.)  │
     ├── Run agent logic (intent → RAG → tool)               │
     ├── Save updated state back to Redis                    │
     └── POST reply → WhatsApp Business API ───────────────►┘
```

### Step-by-Step Integration

**1. Register a WhatsApp Business App**
- Go to [Meta for Developers](https://developers.facebook.com)
- Create an app → Add WhatsApp product
- Get your `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`

**2. Set Up a Webhook Endpoint**
```python
# FastAPI example
from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    body = await request.json()
    
    # Extract message
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone   = message["from"]
    text    = message["text"]["body"]
    
    # Load user state from Redis (keyed by phone number)
    state = redis.get(phone) or create_new_state()
    
    # Run agent
    reply, updated_state = process_message(text, state)
    
    # Save updated state
    redis.set(phone, updated_state)
    
    # Send reply via WhatsApp API
    await send_whatsapp_message(phone, reply)
    return {"status": "ok"}
```

**3. Send Messages Back**
```python
async def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
```

**4. State Persistence**
- Use **Redis** to store `AgentState` per user (keyed by phone number)
- Serialize state as JSON with `json.dumps()` / `json.loads()`
- Set a TTL (e.g., 24 hours) to auto-expire stale sessions

**5. Webhook Verification**
Meta requires your server to verify the webhook with a challenge token:
```python
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == YOUR_VERIFY_TOKEN:
        return int(params["hub.challenge"])
    return {"error": "Unauthorized"}, 403
```

**6. Deploy**
- Host on **Railway**, **Render**, or **AWS EC2**
- Must be HTTPS (Meta requires SSL)
- Use **ngrok** for local testing

---

## ⚠️ Important Notes

### API Quota Limits

The Google Generative AI **free tier** has the following limits:
- **20 requests/day** per model (gemini-2.5-flash)
- **Resets daily at UTC midnight**

For production use, upgrade to a **paid plan** for higher quotas.

### Environment Variables

Make sure to set `GOOGLE_API_KEY` before running:
```bash
$env:GOOGLE_API_KEY="your_key_here"  # PowerShell
python agent.py
```

---

## 🌐 Deployment

### Local Development
Simply run `python agent.py` after setting your API key.

### Production (WhatsApp / Web)
See the [WhatsApp Deployment](#-whatsapp-deployment-via-webhooks) section above for Redis + FastAPI integration patterns.

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py            # Main agent logic (state machine + all nodes)
├── knowledge_base.json # RAG knowledge base (pricing, policies, FAQ)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧩 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM | Google Gemini 2.5 Flash |
| Orchestration | LangGraph (StateGraph + MemorySaver) |
| Framework | LangChain (for LLM + tools) |
| RAG | JSON knowledge base injected into system prompt |
| State | In-memory TypedDict with MemorySaver (Redis for production) |
| Lead Tool | `mock_lead_capture()` function |

---

## ✅ Evaluation Checklist

- [x] Intent classification (3 categories)
- [x] RAG-powered knowledge retrieval
- [x] Stateful conversation (5–6+ turns)
- [x] Progressive lead collection (name → email → platform)
- [x] Tool fires only when ALL fields collected
- [x] Clean, modular code structure
- [x] WhatsApp deployment guide

---

*Built by Gaurav for the ServiceHive / Inflx ML Intern Assignment*
