"""Client LLM unifié : génération simple et conversation multi-tours."""

from __future__ import annotations

import os
import re
import time
from typing import List, Dict, Optional

import requests

GENERAL_SYSTEM_PROMPT = """Vous êtes un assistant conversationnel utile, clair et précis.

Règles :
1. Répondez directement à la question posée, en français sauf si l'utilisateur écrit en anglais.
2. Si vous ne savez pas, dites-le honnêtement — n'inventez pas de faits.
3. Restez concis sauf si l'utilisateur demande des détails.
4. Vous n'avez PAS accès à une base de CVs dans ce mode : ne prétendez pas connaître
   des candidats, scores ou profils spécifiques. Si l'utilisateur pose une question sur
   des CVs ou candidats, suggérez-lui de reformuler en mentionnant « CV », « candidat »
   ou un nom de personne de la base.
5. Tenez compte de l'historique de conversation récent pour rester cohérent."""


def _default_models(provider: str) -> str:
    defaults = {
        "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "openrouter": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        "mistral": "mistral-small-latest",
        "gemini": "gemini-2.0-flash",
    }
    return defaults.get(provider, "llama-3.3-70b-versatile")


class DummyLLM:
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return (
            "⚠️ Aucune clé API configurée. Ajoutez GROQ_API_KEY (ou autre) dans votre fichier .env."
        )

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return self.generate(last_user)


class RealLLM:
    def __init__(self, provider: str, client, model_name: str):
        self.provider = provider
        self.client = client
        self.model_name = model_name

    def _call_with_retry(self, call_fn, max_retries: int = 3):
        for attempt in range(max_retries + 1):
            try:
                return call_fn()
            except Exception as e:
                error_str = str(e)
                if any(
                    k in error_str.lower()
                    for k in ("429", "rate limit", "rate_limit", "resource_exhausted")
                ):
                    if attempt < max_retries:
                        wait_match = re.search(r"in\s+([\d.]+)s", error_str)
                        wait_time = (
                            float(wait_match.group(1)) + 1.0 if wait_match else (2 ** attempt) * 5
                        )
                        time.sleep(wait_time)
                        continue
                raise

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_chat(messages)

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "groq":
            def call_fn():
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content

            return self._call_with_retry(call_fn)

        if self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "cv-pipeline-chatbot",
            }
            payload = {"model": self.model_name, "messages": messages, "temperature": 0.3}
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()["choices"][0]["message"]["content"]

        if self.provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {"model": self.model_name, "messages": messages, "temperature": 0.3}
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Mistral HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()["choices"][0]["message"]["content"]

        if self.provider == "gemini":
            import google.generativeai as genai

            system_parts = [m["content"] for m in messages if m.get("role") == "system"]
            conv_parts = [m for m in messages if m.get("role") != "system"]
            system_instruction = "\n\n".join(system_parts) if system_parts else None
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_instruction,
            )
            history = []
            last_user = None
            for m in conv_parts:
                if m["role"] == "user":
                    last_user = m["content"]
                elif m["role"] == "assistant" and last_user is not None:
                    history.append({"role": "user", "parts": [last_user]})
                    history.append({"role": "model", "parts": [m["content"]]})
                    last_user = None
            chat = model.start_chat(history=history)
            final_user = conv_parts[-1]["content"] if conv_parts else ""
            if conv_parts and conv_parts[-1]["role"] == "user":
                resp = chat.send_message(final_user)
                return resp.text
            return "[Impossible de générer]"

        return "[Provider non supporté]"


def get_llm(provider: str):
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        return DummyLLM()

    model_name = _default_models(provider)

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
    elif provider in ("openrouter", "mistral"):
        client = None
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        client = genai
    else:
        raise ValueError(f"Provider inconnu : {provider}")

    return RealLLM(provider, client, model_name)


def build_chat_messages(
    history: List[Dict],
    system_prompt: str,
    max_turns: int = 8,
) -> List[Dict[str, str]]:
    """Construit la liste de messages API à partir de l'historique Streamlit."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    recent = history[-max_turns * 2 :] if history else []
    for msg in recent:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages
