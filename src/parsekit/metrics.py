import pickle
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .sentence import ConlluSentence


PTB_PUNCT_TAGS = {".", "``", "''", ":", ","}
PUNCT_TAGS = {"pu", "punct"}


def is_punctuation(pos: str) -> bool:
    return pos in PTB_PUNCT_TAGS or pos.lower() in PUNCT_TAGS


def normalize_deprel(rel: str | None, use_deprel_subtypes: bool) -> str | None:
    if rel is None or use_deprel_subtypes:
        return rel
    return rel.split(":", 1)[0]


class DependencyScores:
    total_sentences: Counter[int]
    total_tokens: Counter[int]
    correct_sentences: Counter[int]
    correct_heads: Counter[int]
    correct_labels: Counter[int]
    correct_arcs: Counter[int]
    correct_roots: Counter[int]

    def __init__(self, use_deprel_subtypes: bool = False, ignore_punct: bool = False) -> None:
        self.use_deprel_subtypes = use_deprel_subtypes
        self.ignore_punct = ignore_punct
        self.reset()

    def __str__(self) -> str:
        return self.format()

    def format(self, mode: str = "min") -> str:
        if mode == "min":
            return f"UAS: {self.uas:.2f}%, LAS: {self.las:.2f}%"
        if mode == "full":
            return f"UAS: {self.uas:.2f}%, LAS: {self.las:.2f}%, LS: {self.ls:.2f}%, EM: {self.em:.2f}%, RA: {self.ra:.2f}%"
        raise ValueError(f"Invalid mode: {mode}")

    def format_grouped(self) -> str:
        lines = []
        for length in sorted(self.total_sentences):
            uas = self._pct(self.correct_heads, self.total_tokens, length)
            las = self._pct(self.correct_arcs, self.total_tokens, length)
            lines.append(f"{length}: UAS: {uas:.2f}%, LAS: {las:.2f}%")
        return "\n".join(lines)

    def format_filtered(self, func: Callable[[int], bool] | None = None) -> str:
        if func is None:
            return self.format()
        correct_heads = Counter()
        correct_arcs = Counter()
        total_tokens = Counter()
        for length in sorted(self.total_sentences):
            if func(length):
                correct_heads[length] = self.correct_heads[length]
                correct_arcs[length] = self.correct_arcs[length]
                total_tokens[length] = self.total_tokens[length]
        return f"UAS: {self._pct(correct_heads, total_tokens):.2f}%, LAS: {self._pct(correct_arcs, total_tokens):.2f}%"

    def _pct(self, num: Counter[int], den: Counter[int], length: int | None = None) -> float:
        try:
            if length is None:
                return num.total() / den.total() * 100
            return num[length] / den[length] * 100
        except ZeroDivisionError:
            return float("nan")

    @property
    def uas(self) -> float:
        return self._pct(self.correct_heads, self.total_tokens)

    @property
    def las(self) -> float:
        return self._pct(self.correct_arcs, self.total_tokens)

    @property
    def ls(self) -> float:
        return self._pct(self.correct_labels, self.total_tokens)

    @property
    def em(self) -> float:
        return self._pct(self.correct_sentences, self.total_sentences)

    @property
    def ra(self) -> float:
        return self._pct(self.correct_roots, self.total_sentences)

    def reset(self) -> None:
        self.total_sentences = Counter()
        self.total_tokens = Counter()
        self.correct_sentences = Counter()
        self.correct_heads = Counter()
        self.correct_labels = Counter()
        self.correct_arcs = Counter()
        self.correct_roots = Counter()

    def update(self, sentences: list[ConlluSentence]) -> None:
        assert isinstance(sentences, list)
        for sentence in sentences:
            length = len(sentence)
            all_correct = True
            root_correct = False
            evaluated_tokens = 0
            self.total_sentences[length] += 1
            for token in sentence:
                if self.ignore_punct and is_punctuation(token.pos):
                    continue

                evaluated_tokens += 1
                pred_rel = normalize_deprel(token.pred_rel, self.use_deprel_subtypes)
                gold_rel = normalize_deprel(token.rel, self.use_deprel_subtypes)
                head_correct = token.pred_head == token.head
                label_correct = pred_rel == gold_rel

                if head_correct and label_correct:
                    self.correct_arcs[length] += 1
                else:
                    all_correct = False
                if head_correct:
                    self.correct_heads[length] += 1
                    if token.head == sentence.root.id:
                        root_correct = True
                if label_correct:
                    self.correct_labels[length] += 1
            self.total_tokens[length] += evaluated_tokens
            self.correct_sentences[length] += int(all_correct)
            self.correct_roots[length] += int(root_correct)

    def save(self, path: Path) -> None:
        stats = {
            "use_deprel_subtypes": self.use_deprel_subtypes,
            "ignore_punct": self.ignore_punct,
            "total_sentences": self.total_sentences,
            "total_tokens": self.total_tokens,
            "correct_sentences": self.correct_sentences,
            "correct_heads": self.correct_heads,
            "correct_labels": self.correct_labels,
            "correct_arcs": self.correct_arcs,
            "correct_roots": self.correct_roots,
        }
        with path.open("wb") as f:
            pickle.dump(stats, f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            stats = pickle.load(f)
        self.use_deprel_subtypes = stats.get("use_deprel_subtypes", False)
        self.ignore_punct = stats.get("ignore_punct", False)
        self.total_sentences = stats["total_sentences"]
        self.total_tokens = stats["total_tokens"]
        self.correct_sentences = stats["correct_sentences"]
        self.correct_heads = stats["correct_heads"]
        self.correct_labels = stats["correct_labels"]
        self.correct_arcs = stats["correct_arcs"]
        self.correct_roots = stats["correct_roots"]
