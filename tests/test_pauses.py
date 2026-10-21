"""Test Lab pauses: break tags and punctuation pauses as speech / pause events."""

from app.lab.pauses import DEFAULTS, clean_settings, parse

ON = {**DEFAULTS, "enabled": True}


def test_off_by_default_text_is_unchanged():
    assert parse("Hello. World.", DEFAULTS) == [("speech", "Hello. World.")]


def test_break_tags_always_work_and_are_never_spoken():
    assert parse('One.<break time="2.5s"/> Two.', DEFAULTS) == [("speech", "One."), ("pause", 2500), ("speech", "Two.")]
    assert parse('A<break time="250ms"/>B', DEFAULTS) == [("speech", "A"), ("pause", 250), ("speech", "B")]
    assert parse('Hi <BREAK TIME="1s" /> there', DEFAULTS) == [("speech", "Hi"), ("pause", 1000), ("speech", "there")]
    # a malformed tag is kept as words, never spoken as markup
    assert parse('Hi <break time=soon> there', DEFAULTS) == [("speech", "Hi break time=soon there")]


def test_sentence_pauses_when_enabled():
    assert parse("Hello. World.", ON) == [("speech", "Hello."), ("pause", 350), ("speech", "World.")]
    assert parse("Wait... what? Okay!", ON) == [
        ("speech", "Wait..."), ("pause", 950), ("speech", "what?"), ("pause", 450), ("speech", "Okay!")]
    assert parse("Wait… what?", ON)[1] == ("pause", 950)          # unicode ellipsis, same setting


def test_runs_decimals_and_abbreviations():
    assert parse("Really?! Yes.", ON) == [("speech", "Really?!"), ("pause", 450), ("speech", "Yes.")]  # one pause, the longest
    assert parse("Price is 3.14 dollars.", ON) == [("speech", "Price is 3.14 dollars.")]
    assert parse("Dr. Smith is in. Go.", ON) == [("speech", "Dr. Smith is in."), ("pause", 350), ("speech", "Go.")]
    assert parse("See example.com now.", ON) == [("speech", "See example.com now.")]


def test_explicit_break_adds_to_the_punctuation_pause():
    assert parse('Hello there. <break time="1s"/> Next.', ON) == [
        ("speech", "Hello there."), ("pause", 1350), ("speech", "Next.")]


def test_commas_only_when_set_and_paragraphs():
    assert parse("Hi, there.", ON) == [("speech", "Hi, there.")]
    assert parse("Hi, there.", {**ON, "comma": 200}) == [("speech", "Hi,"), ("pause", 200), ("speech", "there.")]
    assert parse("One\n\nTwo", ON) == [("speech", "One"), ("pause", 650), ("speech", "Two")]


def test_limits():
    assert clean_settings({"period": -5, "comma": 99999, "enabled": 1}) == {**DEFAULTS, "period": 0, "comma": 1000, "enabled": True}
    events = parse('a <break time="99s"/> b', DEFAULTS)
    assert events[1] == ("pause", 10000)                          # one pause: 10 s at most
    many = " ".join(f'w{i} <break time="9s"/>' for i in range(20)) + " end"
    assert sum(v for k, v in parse(many, DEFAULTS) if k == "pause") <= 60000
