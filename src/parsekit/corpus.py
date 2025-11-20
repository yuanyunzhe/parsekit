from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
import xml.etree.ElementTree as ET

from .sentence import ConlluSentence
from .graph import TigerGraph


class ConlluCorpus:
    def __init__(self, sentences: list[ConlluSentence]):
        self.sentences = sentences

    @classmethod
    def from_file(cls, path: Path, projective_only: bool = True) -> ConlluCorpus:
        sentences = []
        with path.open("r", encoding="utf-8") as f:
            lines = []
            for line in f:
                if line.strip():
                    lines.append(line.strip())
                elif lines:
                    sentence = ConlluSentence.from_lines(lines)
                    if not projective_only or sentence.projective:
                        sentences.append(sentence)
                    lines = []
        if lines:
            sentence = ConlluSentence.from_lines(lines)
            if not projective_only or sentence.projective:
                sentences.append(sentence)
        return cls(sentences)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> ConlluSentence:
        return self.sentences[index]

    def __iter__(self) -> Iterator[ConlluSentence]:
        return iter(self.sentences)

    @cached_property
    def tag_set(self) -> set[str]:
        tags = set()
        for sentence in self.sentences:
            for token in sentence:
                tags.add(token.tag)
        return tags

    @cached_property
    def rel_set(self) -> set[str]:
        rels = set()
        for sentence in self.sentences:
            for token in sentence:
                rels.add(token.rel)
        return rels

    @cached_property
    def max_sen_len(self) -> int:
        return max(len(sentence) for sentence in self.sentences)


class TigerCorpus:
    def __init__(self, graphes: list[TigerGraph]):
        self.graphes = graphes

    @classmethod
    def from_file(cls, path: Path) -> TigerCorpus:
        tree = ET.parse(path)
        root = tree.getroot()
        sentences = root.findall("body")[0].findall("s")
        graphes = []
        for sentence in sentences:
            graph = TigerGraph(sentence)
            graphes.append(graph)
        return cls(graphes)

    def __len__(self) -> int:
        return len(self.graphes)

    def __getitem__(self, index: int) -> TigerGraph:
        return self.graphes[index]

    def __iter__(self) -> Iterator[TigerGraph]:
        return iter(self.graphes)
