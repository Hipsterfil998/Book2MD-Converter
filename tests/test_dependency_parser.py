"""Tests for DependencyParser (no Stanza models needed)."""
from book2md.parsing.parser import DependencyParser


class MockWord:
    def __init__(self, id=1, text="Hello", lemma="hello", upos="NOUN",
                 xpos="NN", feats=None, head=0, deprel="root"):
        self.id = id; self.text = text; self.lemma = lemma
        self.upos = upos; self.xpos = xpos; self.feats = feats
        self.head = head; self.deprel = deprel

class MockSentence:
    def __init__(self, words=None): self.words = words or [MockWord()]

class MockDoc:
    def __init__(self, sentences=None): self.sentences = sentences or [MockSentence()]


parser = DependencyParser.__new__(DependencyParser)


class TestMdToTxt:
    def test_strips_heading_markers(self):
        txt = parser._md_to_txt("# Heading")
        assert "#" not in txt and "Heading" in txt

    def test_strips_bold_markers(self):
        txt = parser._md_to_txt("**bold** text")
        assert "**" not in txt and "bold" in txt

    def test_preserves_plain_text(self):
        assert "Hello world" in parser._md_to_txt("Hello world")

    def test_empty_input_returns_empty(self):
        assert parser._md_to_txt("").strip() == ""


class TestDocToConllu:
    def test_tab_separated_10_fields(self):
        line = parser._doc_to_conllu(MockDoc()).strip().split("\n")[0]
        assert len(line.split("\t")) == 10

    def test_field_values(self):
        word = MockWord(id=2, text="Test", lemma="test", upos="NOUN", xpos="NN", head=1, deprel="nsubj")
        fields = parser._doc_to_conllu(MockDoc([MockSentence([word])])).strip().split("\n")[0].split("\t")
        assert fields[0] == "2" and fields[1] == "Test" and fields[2] == "test"
        assert fields[3] == "NOUN" and fields[6] == "1" and fields[7] == "nsubj"

    def test_none_feats_become_underscore(self):
        fields = parser._doc_to_conllu(MockDoc()).strip().split("\n")[0].split("\t")
        assert fields[5] == "_"

    def test_sentences_separated_by_blank_line(self):
        assert "\n\n" in parser._doc_to_conllu(MockDoc([MockSentence(), MockSentence()]))


class TestDocToJson:
    def test_top_level_structure(self):
        result = parser._doc_to_json(MockDoc(), "test.md", "it")
        assert result["file"] == "test.md" and result["lang"] == "it" and "sentences" in result

    def test_sentence_has_tokens_key(self):
        assert "tokens" in parser._doc_to_json(MockDoc(), "f.md", "de")["sentences"][0]

    def test_token_fields(self):
        word = MockWord(id=1, text="Ciao", lemma="ciao", upos="INTJ", xpos="I", feats="X", head=0, deprel="root")
        token = parser._doc_to_json(MockDoc([MockSentence([word])]), "f.md", "it")["sentences"][0]["tokens"][0]
        assert token["text"] == "Ciao" and token["lemma"] == "ciao"
        assert token["upos"] == "INTJ" and token["deprel"] == "root" and token["head"] == 0

    def test_multiple_sentences(self):
        assert len(parser._doc_to_json(MockDoc([MockSentence(), MockSentence()]), "f.md", "it")["sentences"]) == 2
