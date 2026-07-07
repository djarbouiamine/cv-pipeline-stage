"""
quota_tracker.py
Suivi local des quotas journaliers par provider, pour éviter d'appeler
un provider déjà épuisé (au lieu d'attendre l'échec en live).

Reset automatique à minuit (nouvelle journée = compteurs remis à zéro).
"""

import json
import os
from datetime import date

QUOTA_FILE = "output/.quota_state.json"

# Limites journalières par provider — AJUSTE ces valeurs selon
# les vrais plans gratuits que tu utilises.
DEFAULT_LIMITS = {
    "groq": 1000,        # quota gratuit Groq large mais partagé par 3 agents, on reste prudent
    "mistral": 500,
    "openrouter": 50,    # le plus fragile dans tes logs — beaucoup de 429, valeur basse exprès
    "gemini": 200,        # tier gratuit limité, ajuste si tu passes en payant
}


class QuotaTracker:
    def __init__(self, limits=None, path=QUOTA_FILE):
        self.path = path
        self.limits = limits or DEFAULT_LIMITS
        self.state = self._load()

    def _load(self):
        today = str(date.today())
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == today:
                    return data
            except Exception:
                pass
        # Nouveau jour ou fichier absent/corrompu -> reset
        return {"date": today, "counts": {p: 0 for p in self.limits}}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _refresh_if_new_day(self):
        today = str(date.today())
        if self.state.get("date") != today:
            self.state = {"date": today, "counts": {p: 0 for p in self.limits}}
            self._save()

    def can_call(self, provider):
        """True si le provider a encore du quota aujourd'hui."""
        self._refresh_if_new_day()
        limit = self.limits.get(provider)
        if limit is None:
            return True  # provider inconnu = pas de limite suivie, on laisse passer
        used = self.state["counts"].get(provider, 0)
        return used < limit

    def record_call(self, provider):
        """À appeler après CHAQUE appel réussi (ou tenté) à ce provider."""
        self._refresh_if_new_day()
        self.state["counts"][provider] = self.state["counts"].get(provider, 0) + 1
        self._save()

    def mark_exhausted(self, provider):
        """
        À appeler dès qu'on détecte un ProviderExhausted (429 avec long
        Retry-After). Remplit direct le compteur au max pour ce provider,
        pour ne plus jamais retenter aujourd'hui.
        """
        self._refresh_if_new_day()
        limit = self.limits.get(provider, 0)
        self.state["counts"][provider] = limit
        self._save()

    def get_available_provider(self, providers_ordered):
        """
        Prend une liste de providers dans l'ordre de préférence,
        retourne le premier qui a encore du quota, sinon None
        (= tous épuisés aujourd'hui pour cette liste).
        """
        for p in providers_ordered:
            if self.can_call(p):
                return p
        return None

    def status(self):
        """Petit résumé lisible pour debug/logs."""
        self._refresh_if_new_day()
        lines = []
        for p, limit in self.limits.items():
            used = self.state["counts"].get(p, 0)
            lines.append(f"{p}: {used}/{limit}")
        return " | ".join(lines)


# Instance partagée, à importer directement ailleurs :
# from quota_tracker import quota
quota = QuotaTracker()


if __name__ == "__main__":
    # Petit test manuel
    print("État initial:", quota.status())
    quota.record_call("groq")
    quota.record_call("groq")
    print("Après 2 appels groq:", quota.status())
    print("Groq dispo ?", quota.can_call("groq"))