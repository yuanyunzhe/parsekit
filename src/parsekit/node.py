from __future__ import annotations

from dataclasses import dataclass
from xml.etree.ElementTree import Element

from .token import ConstToken


@dataclass(slots=True)
class Node:
    id: str
    parent: Node | None


@dataclass(slots=True)
class Terminal(Node):
    token: ConstToken

    def __str__(self) -> str:
        return f"T({self.id}): {self.token.word}/{self.token.pos}"

    @classmethod
    def from_tiger_xml(cls, element: Element) -> Terminal:
        id = element.attrib["id"]
        token = ConstToken.from_tiger_xml(element)
        return cls(id=id, parent=None, token=token)


@dataclass(slots=True)
class NonTerminal(Node):
    cat: str
    children: list[tuple[Node, str]]

    def __str__(self) -> str:
        return f"NT({self.id}): {self.cat} -> {[f'{label} {child.id}' for child, label in self.children]}"

    @classmethod
    def from_tiger_xml(cls, element: Element) -> NonTerminal:
        id = element.attrib["id"]
        cat = element.attrib["cat"]
        return cls(id=id, parent=None, cat=cat, children=[])

    def add_child(self, child: Node, label: str) -> None:
        self.children.append((child, label))
        child.parent = self
