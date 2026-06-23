# 🤖 Custom AI Chatbot with Memory
### DecodeLabs — Generative AI · Project 1 · Batch 2026

A stateful conversational AI terminal that transforms a stateless LLM into a fully contextual chatbot by engineering an **Artificial Memory Loop** using in-memory session state.

---

## 🧠 Core Architecture

```
Input (Mt ∪ Ht-1) ──▶ GenAI SDK Cloud ──▶ Output (Rt) ──┐
        ▲                                                  │
        └──────────── Append Rt to Ht ────────────────────┘

Mt   = Current user message at turn t
Ht-1 = Historical array of all message transactions up to turn t-1
Rt   = Generated, context-aware model response
```

### Three Core Blocks
| Block | Role |
|---|---|
| **BLOCK 001: Context State** | In-memory `list` array storing full conversation history |
| **BLOCK 002: Query History** | Full `Ht-1` array transmitted with every API call |
| **BLOCK 003: Session Metadata** | Session UUID, turn counter, token estimate |

---

## 🔑 Key Engineering Concepts

### The Amnesiac Cloud Problem
LLMs are **stateless completion engines** — every request is a completely isolated transaction. This project solves that by maintaining state client-side.

### Terminal Append Sequence
Every conversational turn executes exactly **two steps**:
1. **Ingest & Append** — add user message to history as a structured role-content object
2. **Transmit & Record** — send full history array to API, then append model response

```python
# Step 1: Append user message
history.append({"role": "user", "content": user_msg})

# Step 2: Transmit full history + record response
response = client.chat.completions.create(model=model, messages=SYSTEM + history)
history.append({"role": "assistant", "content": response.choices[0].message.content})
```

### Structural Validation Gate
Prevents `400 Bad Request` crashes from empty payloads:
```python
def validate_input(text: str) -> bool:
    return bool(text and text.strip())   # Block empty / whitespace-only inputs
```

### Sliding Window Algorithm (FIFO Pruning)
Prevents context window overflow and token budget exhaustion:
```python
# Drop oldest turn pair (FIFO) when history exceeds limit
while len(history) > MAX_HISTORY_TURNS * 2:
    history = history[2:]   # Remove oldest user+assistant pair
```

---

## ⚙️ Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/custom-ai-chatbot.git
cd custom-ai-chatbot

# 2. Install
pip install -r requirements.txt

# 3. Set API key
export OPENAI_API_KEY="your_openai_key"
# or for Claude:
export ANTHROPIC_API_KEY="your_anthropic_key"
```

---

## 🚀 Usage

### OpenAI (GPT-4o)
```bash
python chatbot.py
```

### Anthropic (Claude)
```bash
python chatbot_claude.py
```

### In-session Commands
| Command | Action |
|---|---|
| `quit` | End the session |
| `history` | Print full conversation history |
| `clear` | Wipe history (fresh context) |
| `exam` | Run the Memory Exam (OpenAI only) |

---

## 🧪 Memory Exam (System Audit)

Tests whether the sliding window correctly preserves early context across large token generations:

```
Phase 1 — State Initialization : "My name is Vipin"       → model acknowledges
Phase 2 — Context Distraction  : "Write a poem about tech" → floods context window
Phase 3 — State Extraction     : "What is my name?"        → must respond: "Vipin" ✓
```

Run it with the `exam` command inside the chatbot.

---

## 🗄️ Persistence Tiers (Enterprise Scaling)

The in-memory array works for local prototyping. For production:

| Layer | Technology | Notes |
|---|---|---|
| **Local** | Python `list` in RAM | Wiped on restart — good for dev only |
| **NoSQL** | Cloud Firestore | Hierarchical `conversations/messages` — strict 1MB doc limit |
| **Relational** | PostgreSQL (JSONB) | `session_id UUID` + `message JSONB` — reliable, scalable |
| **Enterprise** | Firebase SQL Connect | Managed Cloud SQL + GraphQL + IAM — horizontally scaled |

---

## 📐 Stateless vs. Stateful Comparison

| | Stateless API | **Stateful Chat Session** |
|---|---|---|
| API Endpoint | Legacy text completion | **Structured chat completions** |
| State Location | Server forgets each request | **Client-side history array** |
| Context Assembly | Raw string concatenation | **Array of role-content objects** |
| Network Overhead | Low | **High — full array every turn** |
| Failure Risk | Minimal | **Context overflow / token exhaustion** |

---

## 📜 License

MIT License — Built for DecodeLabs Industrial Training Kit · Batch 2026
