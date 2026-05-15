from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from typing import overload

from .token import ConlluToken


class ConlluSentence:
    def __init__(self, tokens: list[ConlluToken], metadata: dict[str, str] | None = None):
        self._tokens = (ConlluToken.create_root(), *tokens)
        self.metadata = metadata or {}

    @classmethod
    def from_lines(cls, lines: list[str]) -> ConlluSentence:
        tokens = []
        metadata = {}
        for line in lines:
            if line.startswith("#"):
                key, sep, value = line[1:].strip().partition("=")
                if sep:
                    metadata[key.strip()] = value.strip()
                continue
            token = ConlluToken.from_line(line)
            if token is not None:
                tokens.append(token)
        return cls(tokens, metadata)

    def __len__(self) -> int:
        return len(self.tokens)

    @overload
    def __getitem__(self, index: int) -> ConlluToken: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ConlluToken, ...]: ...

    def __getitem__(self, index: int | slice) -> ConlluToken | tuple[ConlluToken, ...]:
        if isinstance(index, slice):
            return tuple(self._tokens[index])
        return self._tokens[index]

    def __iter__(self) -> Iterator[ConlluToken]:
        return iter(self.tokens)

    def __str__(self) -> str:
        return self.to_conllu()

    @property
    def text(self) -> str:
        return self.metadata.get("text", "")

    @property
    def sent_id(self) -> str:
        return self.metadata.get("sent_id", "")

    @cached_property
    def root(self) -> ConlluToken:
        return self._tokens[0]

    @cached_property
    def tokens(self) -> tuple[ConlluToken, ...]:
        return self._tokens[1:]

    @cached_property
    def words(self) -> list[str]:
        return [token.word for token in self.tokens]

    @cached_property
    def tags(self) -> list[str]:
        return [token.tag for token in self.tokens]

    @cached_property
    def rels(self) -> list[str]:
        return [token.rel for token in self.tokens]

    @cached_property
    def heads(self) -> list[int]:
        return [token.head for token in self.tokens]

    def to_conllu(self) -> str:
        return "\n".join([token.to_conllu() for token in self.tokens])

    def to_simple(self) -> str:
        return "\n".join([token.to_simple() for token in self.tokens])

    def to_eval(self) -> str:
        return "\n".join([token.to_eval() for token in self.tokens])

    @cached_property
    def projective(self) -> bool:
        arcs = [(token.head, token.id) for token in self._tokens]
        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                x1, y1 = min(arcs[i]), max(arcs[i])
                x2, y2 = min(arcs[j]), max(arcs[j])
                if x1 < x2 < y1 < y2 or x2 < x1 < y2 < y1:
                    return False
        return True

    @property
    def pred_valid(self) -> bool:
        visited = set()
        queue = [self._tokens[0]]
        while queue:
            token = queue.pop(0)
            visited.add(token.id)
            children = [t for t in self._tokens if t.pred_head == token.id and t.id not in visited]
            queue.extend(children)
        return len(visited) == len(self._tokens)

    def reset_predictions(self) -> None:
        for token in self:
            token.reset_prediction()
