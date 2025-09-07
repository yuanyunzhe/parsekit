# parsekit
## Overview
A basic toolkit for dependency parsing: CoNLL‑U I/O, typed data structures, vocabularies, and evaluation metrics.

## Installation
- Local install
    - pip: `pip install -e .`
    - uv: `uv pip install -e .`
- Requires `Python >= 3.12`

## Quick Start

```python
from pathlib import Path
from parsekit.token import ConlluToken
from parsekit.sentence import ConlluSentence
from parsekit.corpus import ConlluCorpus
from parsekit.vocabulary import VocabularySet
from parsekit.metrics import DependencyScores

# Build tokens/sentences
line = "1\tI\t_\tPRON\t_\t_\t2\tnsubj\t_\t_"
token = ConlluToken.from_line(line)
sentence = ConlluSentence.from_lines([line, "2\tam\t_\tAUX\t_\t_\t0\troot\t_\t_"])
print(sentence.to_conllu())

# Load a CoNLL‑U corpus
corpus = ConlluCorpus.from_file(Path("train.conllu"))
vocabs = VocabularySet(corpus)
word_id = vocabs.word_vocab["I"]

# Evaluate
scores = DependencyScores()
scores.update([sentence])
print(scores.format("min"))
```
