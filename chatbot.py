"""
Custom AI Chatbot with Memory
DecodeLabs — Generative AI Project 1
--------------------------------------
Architecture: Stateful Conversational Agent
  Block 001: Context State   — in-memory conversation history array
  Block 002: Query History   — full historical array sent every turn
  Block 003: Session Metadata — session ID, turn counter, token estimate

Memory Loop: Input (Mt ∪ Ht-1) → Process (GenAI SDK) → Output (Rt)
  Mt    : Current user message at turn t
  Ht-1  : Historical array of all message transactions up to turn t-1
  Rt    : Generated, context-aware model response

Safeguards implemented:
  • Structural Validation Gate  — blocks empty / whitespace-only payloads (prevents 400 Bad Request)
  • Sliding Window Algorithm    — FIFO pruning keeps history within token budget
  • Session Metadata tracking   — turn count, estimated token usage
"""

import os
import uuid
import time
from datetime import datetime
from openai import OpenAI                   # pip install openai
# from anthropic import Anthropic           # Uncomment to use Claude instead


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# ── Model settings ──
DEFAULT_MODEL       = "gpt-4o"              # OpenAI model
# DEFAULT_MODEL     = "claude-sonnet-4-6"   # Anthropic model (swap client below)

MAX_HISTORY_TURNS   = 20    # Max message pairs before FIFO pruning
MAX_TOKENS_ESTIMATE = 3000  # Soft token budget guard (rough char/4 estimate)
SYSTEM_PROMPT       = (
    "You are a helpful, intelligent AI assistant. "
    "You remember everything the user has told you in this session. "
    "Be concise, accurate, and friendly."
)

# ── Colors for terminal UI ──
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    BLUE   = "\033[94m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"


# ─────────────────────────────────────────────
#  BLOCK 003: SESSION METADATA
# ─────────────────────────────────────────────

class SessionMetadata:
    def __init__(self):
        self.session_id   = str(uuid.uuid4())[:8].upper()
        self.started_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.turn_count   = 0
        self.total_tokens = 0   # rough estimate

    def increment(self, user_msg: str, assistant_msg: str):
        self.turn_count += 1
        # Rough token estimate: chars / 4
        self.total_tokens += (len(user_msg) + len(assistant_msg)) // 4

    def display(self):
        print(
            f"{C.GRAY}  [SESSION {self.session_id}]"
            f"  Turn: {self.turn_count}"
            f"  ~Tokens used: {self.total_tokens}"
            f"  Started: {self.started_at}{C.RESET}"
        )


# ─────────────────────────────────────────────
#  STRUCTURAL VALIDATION GATE
# ─────────────────────────────────────────────

def validate_input(user_input: str) -> bool:
    """
    Gate: Block empty or whitespace-only payloads.
    Passing these to the LLM API returns a 400 Bad Request and crashes the script.
    """
    return bool(user_input and user_input.strip())


# ─────────────────────────────────────────────
#  SLIDING WINDOW ALGORITHM (FIFO Pruning)
# ─────────────────────────────────────────────

def apply_sliding_window(history: list, max_turns: int, max_token_estimate: int) -> list:
    """
    FIFO pruning to prevent context window overflow and token budget exhaustion.

    Rules:
    1. Never exceed MAX_HISTORY_TURNS message pairs (2 messages per turn).
    2. If estimated token count exceeds budget, drop the oldest turn pair.
    3. Always preserve the most recent context (newest messages stay).

    The history list contains alternating user/assistant message dicts.
    A "turn pair" = 2 items (user msg + assistant msg).
    """
    max_messages = max_turns * 2  # each turn = user + assistant

    # Rule 1: Hard turn limit
    while len(history) > max_messages:
        history = history[2:]   # Drop oldest pair (FIFO)
        print(f"{C.GRAY}  [WINDOW] Pruned oldest turn — history now {len(history)//2} pairs{C.RESET}")

    # Rule 2: Soft token budget guard
    estimated_tokens = sum(len(m["content"]) for m in history) // 4
    while estimated_tokens > max_token_estimate and len(history) >= 2:
        history = history[2:]
        estimated_tokens = sum(len(m["content"]) for m in history) // 4
        print(f"{C.GRAY}  [WINDOW] Token budget guard — pruned to ~{estimated_tokens} tokens{C.RESET}")

    return history


# ─────────────────────────────────────────────
#  BLOCK 001 + 002: MEMORY LOOP CORE
# ─────────────────────────────────────────────

def chat_turn(
    client: OpenAI,
    history: list,      # Ht-1: full history up to previous turn
    user_msg: str,      # Mt: current user message
    model: str,
) -> str:
    """
    Terminal Append Sequence — two steps every turn:
      Step 1 — Ingest & Append:  append user message to history as structured role-content object
      Step 2 — Transmit & Record: send full history to API, append model response to history

    Returns the assistant's response text (Rt).
    """
    # ── STEP 1: Ingest & Append ──
    history.append({"role": "user", "content": user_msg})

    # ── STEP 2: Transmit full history payload ──
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
    )

    assistant_msg = response.choices[0].message.content

    # ── Record: Append model response to history ──
    history.append({"role": "assistant", "content": assistant_msg})

    return assistant_msg


# ─────────────────────────────────────────────
#  MEMORY EXAM (Self-Test)
# ─────────────────────────────────────────────

def run_memory_exam(client: OpenAI, model: str):
    """
    System Audit: The Memory Exam (from blueprint)
      Phase 1 — State Initialization : 'My name is Vipin'   → acknowledgment
      Phase 2 — Context Distraction  : 'Write a poem about tech' → large token generation
      Phase 3 — State Extraction     : 'What is my name?'   → must return 'Vipin'
    """
    print(f"\n{C.YELLOW}{'─'*55}")
    print("  RUNNING MEMORY EXAM (System Audit)")
    print(f"{'─'*55}{C.RESET}")

    history = []

    turns = [
        ("My name is Vipin.",        "Phase 1 — State Initialization"),
        ("Write a short poem about technology and the future.", "Phase 2 — Context Distraction"),
        ("What is my name?",         "Phase 3 — State Extraction (CRITICAL)"),
    ]

    for msg, phase in turns:
        print(f"\n{C.CYAN}[{phase}]{C.RESET}")
        print(f"{C.BLUE}  User : {msg}{C.RESET}")
        history = apply_sliding_window(history, MAX_HISTORY_TURNS, MAX_TOKENS_ESTIMATE)
        response = chat_turn(client, history, msg, model)
        print(f"{C.GREEN}  AI   : {response[:200]}{'…' if len(response)>200 else ''}{C.RESET}")
        time.sleep(0.5)

    # Verify memory
    if "vipin" in history[-1]["content"].lower():
        print(f"\n{C.GREEN}  ✓ MEMORY EXAM PASSED — Model correctly recalled 'Vipin'{C.RESET}")
    else:
        print(f"\n{C.RED}  ✗ MEMORY EXAM FAILED — Model lost context{C.RESET}")

    print(f"{C.YELLOW}{'─'*55}{C.RESET}\n")


# ─────────────────────────────────────────────
#  TERMINAL UI
# ─────────────────────────────────────────────

def print_banner(session: SessionMetadata):
    print(f"\n{C.BOLD}{C.CYAN}{'═'*55}")
    print("   CUSTOM AI CHATBOT WITH MEMORY — DecodeLabs P1")
    print(f"{'═'*55}{C.RESET}")
    print(f"{C.GRAY}   Session ID : {session.session_id}")
    print(f"   Model      : {DEFAULT_MODEL}")
    print(f"   Window     : {MAX_HISTORY_TURNS} turns max (FIFO pruning)")
    print(f"   Commands   : 'quit' | 'history' | 'clear' | 'exam'")
    print(f"{'─'*55}{C.RESET}\n")


def print_history(history: list):
    if not history:
        print(f"{C.GRAY}  [No history yet]{C.RESET}")
        return
    print(f"\n{C.YELLOW}  ── Conversation History ({len(history)//2} turns) ──{C.RESET}")
    for i, msg in enumerate(history):
        role  = "You" if msg["role"] == "user" else " AI"
        color = C.BLUE if msg["role"] == "user" else C.GREEN
        print(f"{color}  [{role}] {msg['content'][:120]}{'…' if len(msg['content'])>120 else ''}{C.RESET}")
    print()


# ─────────────────────────────────────────────
#  MAIN CONVERSATION LOOP
# ─────────────────────────────────────────────

def main():
    # ── Init client ──
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(f"{C.RED}[ERROR] Set OPENAI_API_KEY environment variable.{C.RESET}")
        print("        export OPENAI_API_KEY='your_key_here'")
        return

    client  = OpenAI(api_key=api_key)
    session = SessionMetadata()

    # ── BLOCK 001: Initialize in-memory history array ──
    history: list = []   # Ht — grows with every turn

    print_banner(session)

    # ── Main loop ──
    while True:
        try:
            user_input = input(f"{C.BOLD}{C.BLUE}You: {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.GRAY}  Goodbye!{C.RESET}")
            break

        # ── Special commands ──
        if user_input.lower() == "quit":
            print(f"{C.GRAY}  Session ended. Total turns: {session.turn_count}{C.RESET}")
            break
        elif user_input.lower() == "history":
            print_history(history)
            continue
        elif user_input.lower() == "clear":
            history = []
            print(f"{C.YELLOW}  [CLEARED] Conversation history wiped.{C.RESET}\n")
            continue
        elif user_input.lower() == "exam":
            run_memory_exam(client, DEFAULT_MODEL)
            continue

        # ── STRUCTURAL VALIDATION GATE ──
        if not validate_input(user_input):
            print(f"{C.YELLOW}  [GATE] Empty input blocked — please type a message.{C.RESET}\n")
            continue

        # ── SLIDING WINDOW: Prune before appending ──
        history = apply_sliding_window(history, MAX_HISTORY_TURNS, MAX_TOKENS_ESTIMATE)

        # ── MEMORY LOOP: Mt ∪ Ht-1 → GenAI SDK → Rt ──
        try:
            response = chat_turn(client, history, user_input, DEFAULT_MODEL)
            session.increment(user_input, response)
        except Exception as e:
            print(f"{C.RED}  [API ERROR] {e}{C.RESET}\n")
            # Roll back the user message that was appended inside chat_turn
            if history and history[-2]["role"] == "user":
                history = history[:-2]   # remove both user+assistant if partial
            elif history and history[-1]["role"] == "user":
                history = history[:-1]
            continue

        # ── Display response ──
        print(f"{C.GREEN}{C.BOLD}AI : {C.RESET}{C.GREEN}{response}{C.RESET}\n")
        session.display()


if __name__ == "__main__":
    main()
