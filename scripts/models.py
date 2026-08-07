"""
Model battery, corrected against the actual DARPA "Art of Novel
Signals" RFP rather than the reference document's TGN-first ordering.

The RFP's own DP2 evaluation criterion is explicit: proposers must
demonstrate an "in-context-learning forecasting pipeline that operates
over temporal knowledge graphs without retraining." A temporal GNN
(TGN/EvolveGCN) needs gradient training on every rolling fold, which
directly contradicts that requirement -- so it's demoted here to a
labeled comparison point rather than kept as the anchor model.

Real model, real fallback discipline: ICLForecaster attempts a real
local LLM call via Ollama (free, open-source, no API key -- llama3.1:8b
by default, already installed locally) if use_llm=True and an Ollama
server answers at localhost:11434; falls back to a real Anthropic API
call if ANTHROPIC_API_KEY happens to be set and Ollama isn't reachable;
otherwise uses a fixed, non-trained heuristic scorer (documented
clearly below) so the harness still executes end-to-end without any
live model. The fallback is a stand-in for demonstration purposes
only, not a claim that it performs like a real LLM would.

Local LLM calls take several real seconds each (~10s warm, longer cold)
-- fine for a dedicated, smaller ICL-vs-baseline comparison track and
for generating real case-study reasoning traces, but far too slow to
run inside a 1000+-iteration grand search. use_llm defaults to False
for exactly that reason; the grand search leaves it off and uses the
heuristic fallback (itself real, non-trained retrieval, just without
the language-model reasoning step), and a separate smaller script
turns use_llm on.
"""
import json
import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

FEATURES = [
    "n_events_lag1", "n_events_lag2", "n_events_delta",
    "material_conflict_share_lag1", "material_conflict_share_lag2", "material_conflict_share_delta",
    "mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
    "distinct_actors_lag1", "distinct_actors_lag2", "distinct_actors_delta",
    "mean_tone_lag1", "mean_tone_delta",
]


class NaivePersistence:
    """Not a model -- the floor. 'Escalation continues where it's already happening.'"""
    name = "naive_persistence"

    def fit(self, train_df, label_col):
        return self

    def predict_proba(self, test_df, label_col):
        # persistence = last observed escalation label carried forward as this week's rate
        return test_df["escalation_next_1w"].fillna(0.0).clip(0, 1).values


class HistoricalBaseRate:
    """Not a model -- the other floor. Predicts the country's own historical positive rate."""
    name = "historical_base_rate"

    def fit(self, train_df, label_col):
        self.rate_by_country = train_df.groupby("country")[label_col].mean().to_dict()
        self.overall_rate = train_df[label_col].mean()
        return self

    def predict_proba(self, test_df, label_col):
        return test_df["country"].map(self.rate_by_country).fillna(self.overall_rate).values


class LaggedLogisticRegression:
    """Reference doc model 05: the minimum viable baseline every other model must beat."""
    name = "logreg_lagged_counts"

    def fit(self, train_df, label_col):
        X = train_df[["n_events_lag1", "n_events_lag2", "n_events_delta"]].fillna(0)
        y = train_df[label_col].fillna(0)
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, test_df, label_col):
        X = test_df[["n_events_lag1", "n_events_lag2", "n_events_delta"]].fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


class GraphFeatureGBM:
    """Reference doc model 02: gradient-boosted trees on hand-engineered graph features
    (event frequency, quad-class escalation ratios, Goldstein trend, actor-centrality
    deltas). The explainable baseline the ICL forecaster has to beat to earn its keep."""
    name = "gbm_graph_features"

    def fit(self, train_df, label_col):
        X = train_df[FEATURES].fillna(0)
        y = train_df[label_col].fillna(0)
        pos = max(1, y.sum())
        neg = max(1, len(y) - y.sum())
        self.model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0,
        ).fit(X, y)
        return self

    def predict_proba(self, test_df, label_col):
        X = test_df[FEATURES].fillna(0)
        return self.model.predict_proba(X)[:, 1]


class ICLForecaster:
    """
    THE RFP'S ACTUAL REQUIRED APPROACH: in-context learning over the
    temporal knowledge graph, with no gradient training at any point.

    At prediction time, for each (country, week):
      1. Serialize the recent graph state to structured text -- event
         counts by quad-class, Goldstein trend, distinct-actor count,
         tone -- the same information a human analyst reading a graph
         dashboard would see.
      2. Retrieve the k most similar historical country-weeks by cosine
         similarity over the same feature vector (a real, non-trained
         retrieval step -- historical-analog retrieval per the
         reference doc's model 03), and report how often escalation
         actually followed those analogs.
      3. Prompt an LLM for a calibrated probability and a short
         reasoning trace referencing the analogs.
    No model weights are fit at any point -- 'fit' below only builds
    the historical-analog index from the training window, which is
    retrieval bookkeeping, not training.
    """
    name = "icl_temporal_graph"

    def __init__(self, k_analogs=5, verbose_examples=None, use_llm=False):
        self.k_analogs = k_analogs
        self.verbose_examples = verbose_examples if verbose_examples is not None else []
        self.reasoning_log = []
        self.use_llm = use_llm

    def fit(self, train_df, label_col):
        self.index_df = train_df.dropna(subset=FEATURES + [label_col]).copy()
        self.index_features = self.index_df[FEATURES].fillna(0).values
        self.index_mean = self.index_features.mean(axis=0)
        self.index_std = self.index_features.std(axis=0) + 1e-9
        self.index_labels = self.index_df[label_col].values
        return self

    def _retrieve_analogs(self, feature_vec):
        z_query = (feature_vec - self.index_mean) / self.index_std
        z_index = (self.index_features - self.index_mean) / self.index_std
        sims = z_index @ z_query / (np.linalg.norm(z_index, axis=1) * np.linalg.norm(z_query) + 1e-9)
        top_idx = np.argsort(-sims)[: self.k_analogs]
        return top_idx, sims[top_idx]

    def _serialize_context(self, row):
        return (
            f"Country: {row['country']}, week of {row['week']}\n"
            f"Events last week: {row['n_events_lag1']:.0f} (prior week {row['n_events_lag2']:.0f}, "
            f"change {row['n_events_delta']:+.0f})\n"
            f"Material-conflict share: {row['material_conflict_share_lag1']:.2f} "
            f"(change {row['material_conflict_share_delta']:+.2f})\n"
            f"Mean Goldstein score: {row['mean_goldstein_lag1']:.2f} (change {row['mean_goldstein_delta']:+.2f}), "
            f"lower = more conflictual\n"
            f"Distinct actors mentioned: {row['distinct_actors_lag1']:.0f} "
            f"(change {row['distinct_actors_delta']:+.0f})\n"
            f"Average source tone: {row['mean_tone_lag1']:.2f} (change {row['mean_tone_delta']:+.2f}), "
            f"more negative = more hostile coverage"
        )

    def _call_llm(self, context_text, analog_summary):
        if not self.use_llm:
            return None, None, "llm_disabled"

        prompt = (
            "You are a conflict early-warning analyst. Given the current temporal "
            "knowledge graph state for a country-week below, and the outcomes of the "
            "most similar historical weeks retrieved from the graph, output a JSON "
            "object {\"probability\": <0-1 float>, \"reasoning\": \"<one paragraph>\"} "
            "estimating the probability of a material-conflict escalation in the "
            "following week. Do not use any information beyond what's provided.\n\n"
            f"CURRENT STATE:\n{context_text}\n\nHISTORICAL ANALOGS:\n{analog_summary}"
        )

        # Primary path: free, local, open-source Ollama -- no API key,
        # no per-call cost, genuinely runs the reasoning through a real
        # language model instead of the hand-specified fallback formula.
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read())
            text = payload["response"]
            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return float(parsed["probability"]), parsed.get("reasoning", ""), f"live_llm_ollama:{OLLAMA_MODEL}"
        except Exception as e_ollama:
            pass  # fall through to Anthropic if a key happens to be set, else heuristic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
                parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
                return float(parsed["probability"]), parsed.get("reasoning", ""), "live_llm_anthropic"
            except Exception as e:
                return None, f"LLM call failed: ollama={e_ollama}, anthropic={e}", "error"
        return None, f"Ollama unreachable ({e_ollama}), no ANTHROPIC_API_KEY set", "no_llm_available"

    def predict_proba(self, test_df, label_col):
        probs = []
        for _, row in test_df.iterrows():
            feature_vec = row[FEATURES].fillna(0).values.astype(float)
            top_idx, sims = self._retrieve_analogs(feature_vec)
            analog_rows = self.index_df.iloc[top_idx]
            analog_escalation_rate = self.index_labels[top_idx].mean()

            analog_lines = []
            for (_, arow), sim, lab in zip(analog_rows.iterrows(), sims, self.index_labels[top_idx]):
                analog_lines.append(
                    f"  - {arow['country']} {arow['week']} (similarity {sim:.2f}): "
                    f"escalation followed = {bool(lab)}"
                )
            analog_summary = f"{len(top_idx)} nearest analogs, {analog_escalation_rate:.0%} led to escalation:\n" + "\n".join(analog_lines)
            context_text = self._serialize_context(row)

            llm_prob, reasoning, mode = self._call_llm(context_text, analog_summary)
            if llm_prob is not None:
                prob = llm_prob
            else:
                # documented heuristic fallback -- NOT the real ICL mechanism.
                # Fixed, hand-specified logistic-shaped combination of the
                # retrieved-analog rate and the raw feature delta, involving
                # no gradient training of any kind, used only so the harness
                # runs end-to-end without API credentials in this environment.
                delta_signal = (
                    0.5 * np.tanh(row["material_conflict_share_delta"] * 3) +
                    0.3 * np.tanh(-row["mean_goldstein_delta"] / 3) +
                    0.2 * np.tanh(row["distinct_actors_delta"] / 5)
                )
                prob = float(np.clip(0.5 * analog_escalation_rate + 0.5 * (0.5 + 0.5 * delta_signal), 0, 1))
                reasoning = (
                    f"[heuristic fallback, mode={mode}] {analog_escalation_rate:.0%} of retrieved analogs "
                    f"escalated; combined with raw feature deltas -> {prob:.2f}"
                )
            probs.append(prob)
            self.reasoning_log.append({
                "country": row["country"], "week": str(row["week"]),
                "probability": prob, "reasoning": reasoning,
                "top_analogs": analog_summary,
            })
        return np.array(probs)

    def name_(self):
        return self.name


MODEL_REGISTRY = {
    "icl_temporal_graph": ICLForecaster,
    "gbm_graph_features": GraphFeatureGBM,
    "logreg_lagged_counts": LaggedLogisticRegression,
    "naive_persistence": NaivePersistence,
    "historical_base_rate": HistoricalBaseRate,
}
