from __future__ import annotations

from dataclasses import dataclass
from xml.etree.ElementTree import Element

EMPTY = "_"
ROOT = "<ROOT>"
PAD = "<PAD>"
UNK = "<UNK>"

ROOT_ID = 0
ROOT_HEAD = -1


@dataclass(slots=True)
class Token:
    id: int
    word: str
    pos: str


@dataclass(slots=True)
class ConlluToken(Token):
    head: int
    rel: str
    lemma: str | None = EMPTY
    xpos: str | None = EMPTY
    feats: str | None = EMPTY
    deps: str | None = EMPTY
    misc: str | None = EMPTY
    pred_head: int | None = None
    pred_rel: str | None = None

    @classmethod
    def from_line(cls, line: str) -> ConlluToken | None:
        try:
            id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = line.strip().split("\t")
            assert id.isdigit()
            return cls(
                id=int(id),
                word=form,
                lemma=lemma,
                pos=upos,
                xpos=xpos,
                feats=feats,
                head=int(head),
                rel=deprel,
                deps=deps,
                misc=misc,
            )
        except Exception:
            return None

    @classmethod
    def create_root(cls) -> ConlluToken:
        return cls(
            id=ROOT_ID,
            word=ROOT,
            pos=ROOT,
            head=ROOT_HEAD,
            rel=ROOT,
        )

    @property
    def tag(self) -> str:
        return self.pos

    def __str__(self) -> str:
        return self.to_conllu()

    def to_conllu(self) -> str:
        return f"{self.id}\t{self.word}\t{self.lemma}\t{self.pos}\t{self.xpos}\t{self.feats}\t{self.head}\t{self.rel}\t{self.deps}\t{self.misc}"

    def to_simple(self) -> str:
        return f"{self.id} {self.word} {self.tag} {self.head} {self.rel}"

    def to_eval(self) -> str:
        return f"{self.id} {self.word} {self.tag} {self.head} {self.pred_head} {self.rel} {self.pred_rel}"

    def reset_prediction(self) -> None:
        self.pred_head = None
        self.pred_rel = None


@dataclass(slots=True)
class ConstToken(Token):
    @classmethod
    def from_tiger_xml(cls, element: Element) -> ConstToken:
        id = element.attrib["id"].split("_")[-1]
        word = element.attrib["word"]
        pos = element.attrib["pos"]
        assert id.isdigit()
        return cls(id=int(id), word=word, pos=pos)

    def __str__(self) -> str:
        return f"{self.id} {self.word} {self.pos}"
