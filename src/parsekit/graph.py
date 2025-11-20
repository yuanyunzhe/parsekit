from __future__ import annotations

from xml.etree.ElementTree import Element

from .node import Node, NonTerminal, Terminal


class TigerGraph:
    def __init__(self, sentence: Element):
        self.terminals: list[Terminal] = []
        self.nonterminals: list[NonTerminal] = []
        self.id2node: dict[str, Node] = {}
        self._build_graph(sentence)

    def _build_graph(self, sentence: Element) -> None:
        graph = sentence.find("graph")
        assert graph is not None
        for t in graph.findall("terminals/t"):
            terminal = Terminal.from_tiger_xml(t)
            self.terminals.append(terminal)
            self.id2node[terminal.id] = terminal
        for nt in graph.findall("nonterminals/nt"):
            nonterminal = NonTerminal.from_tiger_xml(nt)
            self.nonterminals.append(nonterminal)
            self.id2node[nonterminal.id] = nonterminal
        root_id = graph.attrib["root"]
        self.root = self.id2node[root_id]
        for nt in graph.findall("nonterminals/nt"):
            parent = self.id2node[nt.attrib["id"]]
            for edge in nt.findall("edge"):
                child = self.id2node[edge.attrib["idref"]]
                label = edge.attrib["label"]
                assert isinstance(parent, NonTerminal)
                parent.add_child(child, label)

    def __str__(self) -> str:
        lines = ["Terminals:"]
        for t in self.terminals:
            lines.append(f"  {t}")
        lines.append("NonTerminals:")
        for nt in self.nonterminals:
            lines.append(f"  {nt}")
        return "\n".join(lines)

    def to_bracketed(self) -> str:
        def _rec(node: Node) -> str:
            if isinstance(node, Terminal):
                return f"({node.token.pos} {node.token.word})"
            if isinstance(node, NonTerminal):
                return f"({node.cat} {' '.join(_rec(child) for child, _ in node.children)})"
            raise ValueError("Unknown node type")

        return _rec(self.root)
