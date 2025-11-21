import xml.etree.ElementTree as ET


def tiger_split(tiger_path: str, train_path: str, dev_path: str, test_path: str) -> None:
    """
    Splits a TIGER XML file into train, dev, and test sets.
        Train: 40,472 sentences
        Dev: 5,000 sentences
        Test: 5,000 sentences
    Two sentences, #46234 and #50224 are excluded from the test set because of annotation errors.
    """

    tree = ET.parse(tiger_path)
    root = tree.getroot()
    body = root.find("body")
    sentences = body.findall("s")

    train_sentences = sentences[:40472]
    dev_sentences = sentences[40472:45472]
    test_sentences = sentences[45472:]

    def write_sentences(sentences: list[ET.Element], path: str) -> None:
        new_root = ET.Element("body")
        for s in sentences:
            new_root.append(s)
        new_tree = ET.ElementTree(new_root)
        new_tree.write(path, encoding="utf-8", xml_declaration=True)

    write_sentences(train_sentences, train_path)
    write_sentences(dev_sentences, dev_path)
    write_sentences(test_sentences, test_path)


if __name__ == "__main__":
    tiger_split(
        tiger_path="data/tiger/tiger_release_27-1-2010.xml",
        train_path="data/tiger/train.xml",
        dev_path="data/tiger/dev.xml",
        test_path="data/tiger/test.xml",
    )
