from qtts.text_utils import split_text


def test_split_text_keeps_short_text():
    text = "hello world"
    assert split_text(text, max_chars=20) == ["hello world"]


def test_split_text_splits_long_sentence():
    text = "a" * 50
    parts = split_text(text, max_chars=10)
    assert len(parts) == 5
    assert all(len(p) <= 10 for p in parts)


def test_split_text_sentence_boundary():
    text = "one. two. three."
    parts = split_text(text, max_chars=8)
    assert parts == ["one.", "two.", "three."]
