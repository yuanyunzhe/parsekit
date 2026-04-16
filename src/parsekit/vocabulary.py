from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import overload

from .corpus import ConlluCorpus
from .token import PAD, ROOT, UNK

PAD_IDX = 0
NO_IDX = -1


class VocabularySet:
    def __init__(self, corpus: ConlluCorpus):
        words, tags, rels = [], [], []
        for sentence in corpus:
            words.extend(sentence.words)
            tags.extend(sentence.tags)
            rels.extend(sentence.rels)
        self.word_vocab = Vocabulary("words", words, special_items=[PAD, UNK, ROOT], min_freq=1)
        self.tag_vocab = Vocabulary("tags", tags, special_items=[PAD, UNK, ROOT])
        self.rel_vocab = Vocabulary("rels", rels, special_items=[PAD, UNK, ROOT])

    def save(self, path: Path) -> None:
        data = {
            "word_vocab": self.word_vocab.to_dict(),
            "tag_vocab": self.tag_vocab.to_dict(),
            "rel_vocab": self.rel_vocab.to_dict(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> VocabularySet:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        vocabs = cls.__new__(cls)
        vocabs.word_vocab = Vocabulary.from_dict(data["word_vocab"])
        vocabs.tag_vocab = Vocabulary.from_dict(data["tag_vocab"])
        vocabs.rel_vocab = Vocabulary.from_dict(data["rel_vocab"])
        return vocabs


class Vocabulary:
    def __init__(self, name: str, items: list[str], special_items: list[str], min_freq: int = 1):
        self.name = name
        self.item2idx: dict[str, int] = {}
        self.idx2item: dict[int, str] = {}

        for item in special_items:
            idx = len(self.item2idx)
            self.item2idx[item] = idx
            self.idx2item[idx] = item

        counter = Counter(items)
        for item, count in counter.most_common():
            if count >= min_freq and item not in self.item2idx:
                idx = len(self.item2idx)
                self.item2idx[item] = idx
                self.idx2item[idx] = item

    def __len__(self) -> int:
        return len(self.item2idx)

    @overload
    def __getitem__(self, item: str) -> int: ...

    @overload
    def __getitem__(self, item: list[str]) -> list[int]: ...

    def __getitem__(self, item: str | list[str]) -> int | list[int]:
        if isinstance(item, list):
            return [self.__getitem__(i) for i in item]
        if item in self.item2idx:
            return self.item2idx[item]
        if UNK in self.item2idx:
            return self.item2idx[UNK]
        return 0

    def __contains__(self, item: str) -> bool:
        return item in self.item2idx

    def __iter__(self) -> Iterator[str]:
        return iter(self.item2idx.keys())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "item2idx": self.item2idx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Vocabulary:
        vocab = cls.__new__(cls)
        vocab.name = data.get("name", "")
        vocab.item2idx = {item: int(idx) for item, idx in data["item2idx"].items()}
        vocab.idx2item = {idx: item for item, idx in vocab.item2idx.items()}
        return vocab
