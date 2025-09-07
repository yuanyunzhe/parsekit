from __future__ import annotations

from dataclasses import dataclass

EMPTY = "_"
ROOT = "<ROOT>"
PAD = "<PAD>"
UNK = "<UNK>"

ROOT_ID = 0
ROOT_HEAD = -1


@dataclass(slots=True)
class ConlluToken:
    id: int
    form: str
    upos: str
    head: int
    deprel: str
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
            id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = (
                line.strip().split("\t")
            )
            assert id.isdigit()
            return cls(
                id=int(id),
                form=form,
                lemma=lemma,
                upos=upos,
                xpos=xpos,
                feats=feats,
                head=int(head),
                deprel=deprel,
                deps=deps,
                misc=misc,
            )
        except Exception:
            return None

    @classmethod
    def create_root(cls) -> ConlluToken:
        return cls(
            id=ROOT_ID,
            form=ROOT,
            upos=ROOT,
            head=ROOT_HEAD,
            deprel=ROOT,
        )

    @property
    def word(self) -> str:
        return self.form

    @property
    def tag(self) -> str:
        return self.upos

    @property
    def rel(self) -> str:
        return self.deprel

    def __str__(self) -> str:
        return self.to_conllu()

    def to_conllu(self) -> str:
        return f"{self.id}\t{self.form}\t{self.lemma}\t{self.upos}\t{self.xpos}\t{self.feats}\t{self.head}\t{self.deprel}\t{self.deps}\t{self.misc}"

    def to_simple(self) -> str:
        return f"{self.id} {self.word} {self.tag} {self.head} {self.rel}"

    def to_eval(self) -> str:
        return f"{self.id} {self.word} {self.tag} {self.head} {self.pred_head} {self.rel} {self.pred_rel}"

    def reset_prediction(self) -> None:
        self.pred_head = None
        self.pred_rel = None
