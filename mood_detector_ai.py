"""
Mood-Based AI Agent (Python)

A feature-rich, student-friendly chatbot you can submit as an AIML mini-project.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import uuid
import queue
import random
import string
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# --- Third-party imports ---
try:
    import nltk
except Exception as e:
    print("Installing/initializing NLTK may be required.")
    raise e

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---------------------------
# Section 0: Utilities & Setup
# ---------------------------

NLTK_PACKAGES = [
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
]

def ensure_nltk():
    """Download required NLTK resources if missing."""
    for pkg, path in NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

ensure_nltk()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# File system paths for persistence
RUN_ID = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(os.getcwd(), "chat_artifacts")
LOG_DIR = os.path.join(SAVE_DIR, "logs")
EXPORT_DIR = os.path.join(SAVE_DIR, "exports")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------------------
# Section 1: Data & Knowledge
# ---------------------------

DEFAULT_FAQ: List[Tuple[str, str]] = [
    ("What is AI?", "AI stands for Artificial Intelligence—machines performing tasks that mimic human intelligence."),
    ("What is NLP?", "NLP is Natural Language Processing: enabling computers to understand human language."),
    ("What is sentiment analysis?", "It detects the emotional tone (positive/negative/neutral) of text."),
    ("Who made you?", "I was built by an AIML student using Python, NLTK, TextBlob, and scikit-learn."),
    ("How do you work?", "I detect intent, analyze sentiment, and retrieve answers using a tiny knowledge base."),
    ("What can you do?", "I can chat, detect mood, answer FAQs, remember recent context, and export logs."),
    ("What is machine learning?", "ML lets systems learn patterns from data instead of using explicit rules."),
    ("Explain TF-IDF?", "TF-IDF scores words by importance across documents using term and inverse document frequency."),
    ("How to clear memory?", "Type /clear to wipe the short-term memory for this session."),
]

# ---------------------------
# Section 2: Core Components
# ---------------------------

@dataclass
class ConversationMemory:
    """Maintains a rolling window of recent user utterances for context."""
    max_len: int = 5
    items: List[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.items.append(text)
        if len(self.items) > self.max_len:
            self.items.pop(0)

    def clear(self) -> None:
        self.items.clear()

    def as_context(self) -> str:
        return " | ".join(self.items)


class SentimentAnalyzer:
    """Wrapper around TextBlob polarity for easy swapping later."""

    @staticmethod
    def polarity(text: str) -> float:
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return 0.0

    @staticmethod
    def mood_label(score: float) -> str:
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        return "neutral"


class IntentClassifier:
    """Very simple hybrid intent detection.

    1) Rule-based regex intents (high precision on simple patterns)
    2) Fallback: TF-IDF similarity vs. intent examples
    """

    def __init__(self):
        self.rule_intents: List[Tuple[str, re.Pattern]] = [
            ("greet", re.compile(r"\b(hi|hello|hey|namaste|hola)\b", re.I)),
            ("bye", re.compile(r"\b(bye|goodbye|see ya|see you|exit|quit)\b", re.I)),
            ("help", re.compile(r"\b(help|what can you do|commands|options)\b", re.I)),
            ("thanks", re.compile(r"\b(thanks|thank you|thx|tysm)\b", re.I)),
            ("who", re.compile(r"\bwho (are|r) you\b", re.I)),
            ("what", re.compile(r"\bwhat (can you do|is your name)\b", re.I)),
            ("time", re.compile(r"\b(time|date)\b", re.I)),
        ]

        # Examples for similarity fallback
        self.intent_examples: Dict[str, List[str]] = {
            "greet": ["hi", "hello there", "hey bot", "good morning"],
            "bye": ["bye", "see you", "talk later", "good night"],
            "help": ["help", "what can you do", "show commands"],
            "thanks": ["thanks", "thank you so much", "appreciate it"],
            "who": ["who are you", "introduce yourself"],
            "time": ["what is the time", "tell me the date"],
        }

        corpus = []
        labels = []
        for label, examples in self.intent_examples.items():
            for ex in examples:
                corpus.append(ex)
                labels.append(label)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.X = self.vectorizer.fit_transform(corpus)
        self.labels = labels

    def predict(self, text: str) -> Optional[str]:
        # 1) Rule pass
        for label, pattern in self.rule_intents:
            if pattern.search(text):
                return label
        # 2) Similarity pass
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.X)[0]
        idx = sims.argmax()
        if sims[idx] > 0.35:  # threshold to avoid random matches
            return self.labels[idx]
        return None


class KnowledgeBase:
    """Tiny FAQ retriever using TF-IDF cosine similarity."""

    def __init__(self, pairs: List[Tuple[str, str]]):
        self.questions = [q for q, _ in pairs]
        self.answers = [a for _, a in pairs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.X = self.vectorizer.fit_transform(self.questions)

    def retrieve(self, query: str, top_k: int = 1, min_sim: float = 0.25) -> Optional[str]:
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.X)[0]
        best_i = sims.argmax()
        if sims[best_i] >= min_sim:
            return self.answers[best_i]
        return None

# ---------------------------
# Section 3: Response Engine
# ---------------------------

class ResponseGenerator:
    """Maps intents + sentiment + memory to concrete responses."""

    def __init__(self, kb: KnowledgeBase, memory: ConversationMemory):
        self.kb = kb
        self.memory = memory

    @staticmethod
    def _time_string() -> str:
        now = dt.datetime.now()
        return now.strftime("%A, %d %B %Y, %I:%M %p")

    def reply_for_intent(self, intent: str, sentiment: float) -> str:
        mood = SentimentAnalyzer.mood_label(sentiment)
        if intent == "greet":
            return f"Hello! I'm your mood-aware AI agent. You're sounding {mood} today. How can I help? 😊"
        if intent == "bye":
            return "Goodbye! I saved this chat to logs. See you soon 👋"
        if intent == "help":
            return (
                "Commands: /help, /stats, /clear, /export, /about\n"
                "I can chat, detect your mood, answer mini-FAQs, and remember recent context."
            )
        if intent == "thanks":
            return "You're welcome! Happy to help ✨"
        if intent == "who":
            return "I'm a Python AI agent built for an AIML class—sentiment-aware with tiny retrieval."
        if intent == "time":
            return f"Current local time is {self._time_string()}"
        if intent == "what":
            return "I can chat, analyze your sentiment, and fetch answers from a tiny knowledge base."
        return "I'm not sure I caught that intent. Tell me more."

    def generic_reply(self, user_text: str, sentiment: float) -> str:
        # 1) Try KB FAQ
        kb_ans = self.kb.retrieve(user_text)
        if kb_ans:
            return kb_ans

        # 2) Sentiment-guided small talk
        mood = SentimentAnalyzer.mood_label(sentiment)
        if mood == "positive":
            return (
                "Love the energy! If you're exploring AI, try asking me about NLP or TF-IDF, "
                "or use /help to see commands."
            )
        elif mood == "negative":
            # Use memory context for empathy
            context = self.memory.as_context()
            suffix = f" I remember you said: {context}." if context else ""
            return (
                "I'm sorry you're feeling low. Want to talk about it? I can listen or answer AI questions." + suffix
            )
        else:
            return (
                "Got it. If you ask a question, I can try to retrieve an answer from my tiny knowledge base."
            )

# ---------------------------
# Section 4: Chatbot Orchestrator
# ---------------------------

class MoodAgent:
    """End-to-end agent: I/O loop, logging, routing, and commands."""

    def __init__(self, kb_pairs: List[Tuple[str, str]] = None, memory_len: int = 5):
        self.kb = KnowledgeBase(kb_pairs or DEFAULT_FAQ)
        self.memory = ConversationMemory(max_len=memory_len)
        self.intents = IntentClassifier()
        self.generator = ResponseGenerator(self.kb, self.memory)
        self.session_id = str(uuid.uuid4())[:8]
        self.log_path = os.path.join(LOG_DIR, f"chat_{RUN_ID}_{self.session_id}.txt")
        self.turn_count = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._init_log()

    # ----- Logging -----
    def _init_log(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"SESSION {self.session_id} START {dt.datetime.now().isoformat()}\n")

    def _log_turn(self, user: str, bot: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"U: {user}\nB: {bot}\n---\n")

    # ----- Stats -----
    def _update_stats(self, user: str, bot: str) -> None:
        self.turn_count += 1
        self.tokens_in += len(user.split())
        self.tokens_out += len(bot.split())

    def stats(self) -> str:
        return (
            f"Turns: {self.turn_count}\n"
            f"User words: {self.tokens_in}\n"
            f"Bot words: {self.tokens_out}\n"
            f"Memory size: {len(self.memory.items)}/{self.memory.max_len}\n"
            f"Log file: {self.log_path}"
        )

    # ----- Commands -----
    def handle_command(self, cmd: str) -> str:
        cmd = cmd.strip().lower()
        if cmd == "/help":
            return (
                "Available: /help, /stats, /clear, /export, /about\n"
                "Tip: Ask me about AI basics or just chat."
            )
        if cmd == "/stats":
            return self.stats()
        if cmd == "/clear":
            self.memory.clear()
            return "Cleared short-term memory."
        if cmd == "/about":
            return (
                "MoodAgent v1 — Python (NLTK, TextBlob, scikit-learn).\n"
                "Features: sentiment-aware replies, intents, TF-IDF FAQ, memory, logs."
            )
        if cmd == "/export":
            return self.export()
        return "Unknown command. Try /help"

    def export(self) -> str:
        # Export memory as CSV
        mem_path = os.path.join(EXPORT_DIR, f"memory_{RUN_ID}_{self.session_id}.csv")
        pd.DataFrame({"recent_user_utterances": self.memory.items}).to_csv(mem_path, index=False)
        return f"Exported memory to {mem_path}"

    # ----- Core chat step -----
    def step(self, user_text: str) -> str:
        if not user_text.strip():
            return "Please type something."

        # Commands
        if user_text.strip().startswith("/"):
            bot = self.handle_command(user_text.strip())
            self._log_turn(user_text, bot)
            self._update_stats(user_text, bot)
            return bot

        # Add to memory before responding (enables empathy)
        self.memory.add(user_text)

        # Sentiment
        sent = SentimentAnalyzer.polarity(user_text)

        # Intent
        intent = self.intents.predict(user_text)
        if intent:
            bot = self.generator.reply_for_intent(intent, sent)
        else:
            bot = self.generator.generic_reply(user_text, sent)

        self._log_turn(user_text, bot)
        self._update_stats(user_text, bot)
        return bot

    # ----- CLI Loop -----
    def chat(self) -> None:
        print("\n🤖 Mood-Based AI Agent — Type /help for commands. Type 'bye' to exit.\n")
        while True:
            try:
                user = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting…")
                break

            if user.lower().strip() in {"bye", "exit", "quit"}:
                print("Bot: " + self.generator.reply_for_intent("bye", 0.0))
                break

            bot = self.step(user)
            print("Bot:", bot)

        print(f"\nLog saved at: {self.log_path}")

# ---------------------------
# Section 5: Entry Point
# ---------------------------

if __name__ == "__main__":
    agent = MoodAgent()
    agent.chat()
