"""Tests for DependencyParser (no Stanza models needed)."""
from dependency_parsing.dependency_parsing import DependencyParser


# ── Mock Stanza objects ───────────────────────────────────────────────────────

class MockWord:
    def __init__(self, id=1, text="Hello", lemma="hello", upos="NOUN",
                 xpos="NN", feats=None, head=0, deprel="root"):
        self.id     = id
        self.text   = text
        self.lemma  = lemma
        self.upos   = upos
        self.xpos   = xpos
        self.feats  = feats
        self.head   = head
        self.deprel = deprel


class MockSentence:
    def __init__(self, words=None):
        self.words = words or [MockWord()]


class MockDoc:
    def __init__(self, sentences=None):
        self.sentences = sentences or [MockSentence()]


parser = DependencyParser.__new__(DependencyParser)


# ── Markdown → plain text ─────────────────────────────────────────────────────

class TestMdToTxt:
    def test_strips_heading_markers(self):
        txt = parser._md_to_txt("# Heading")
        assert "#" not in txt
        assert "Heading" in txt

    def test_strips_bold_markers(self):
        txt = parser._md_to_txt("**bold** text")
        assert "**" not in txt
        assert "bold" in txt

    def test_preserves_plain_text(self):
        assert "Hello world" in parser._md_to_txt("Hello world")

    def test_empty_input_returns_empty(self):
        assert parser._md_to_txt("").strip() == ""


# ── Document → CoNLL-U ────────────────────────────────────────────────────────

class TestDocToConllu:
    def test_tab_separated_10_fields(self):
        doc = MockDoc([MockSentence([MockWord(id=1, text="Hallo")])])
        line = parser._doc_to_conllu(doc).strip().split("\n")[0]
        assert len(line.split("\t")) == 10

    def test_field_values(self):
        word = MockWord(id=2, text="Test", lemma="test", upos="NOUN",
                        xpos="NN", feats=None, head=1, deprel="nsubj")
        doc = MockDoc([MockSentence([word])])
        fields = parser._doc_to_conllu(doc).strip().split("\n")[0].split("\t")
        assert fields[0] == "2"       # id
        assert fields[1] == "Test"    # text
        assert fields[2] == "test"    # lemma
        assert fields[3] == "NOUN"    # upos
        assert fields[6] == "1"       # head
        assert fields[7] == "nsubj"   # deprel

    def test_none_feats_become_underscore(self):
        doc = MockDoc([MockSentence([MockWord(feats=None)])])
        fields = parser._doc_to_conllu(doc).strip().split("\n")[0].split("\t")
        assert fields[5] == "_"

    def test_sentences_separated_by_blank_line(self):
        doc = MockDoc([MockSentence(), MockSentence()])
        result = parser._doc_to_conllu(doc)
        assert "\n\n" in result


# ── Document → JSON ───────────────────────────────────────────────────────────

class TestDocToJson:
    def test_top_level_structure(self):
        result = parser._doc_to_json(MockDoc(), "test.md", "it")
        assert result["file"] == "test.md"
        assert result["lang"] == "it"
        assert "sentences" in result

    def test_sentence_has_tokens_key(self):
        result = parser._doc_to_json(MockDoc(), "f.md", "de")
        assert "tokens" in result["sentences"][0]

    def test_token_fields(self):
        word = MockWord(id=1, text="Ciao", lemma="ciao", upos="INTJ",
                        xpos="I", feats="X", head=0, deprel="root")
        result = parser._doc_to_json(MockDoc([MockSentence([word])]), "f.md", "it")
        token = result["sentences"][0]["tokens"][0]
        assert token["text"]   == "Ciao"
        assert token["lemma"]  == "ciao"
        assert token["upos"]   == "INTJ"
        assert token["deprel"] == "root"
        assert token["head"]   == 0

    def test_multiple_sentences(self):
        doc = MockDoc([MockSentence(), MockSentence()])
        result = parser._doc_to_json(doc, "f.md", "it")
        assert len(result["sentences"]) == 2
