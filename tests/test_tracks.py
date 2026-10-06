"""Audio track choice: picked without asking when only one track is in the voice's language."""

from app.voice.freeform import best_track_in, obvious_track


def track(index, language, title="", default=False, channels=2):
    return {"index": index, "language": language, "title": title, "default": default,
            "channels": channels, "surround": channels >= 5}


def test_one_english_track_is_obvious():
    # ALF on DVD: English stereo + Brazilian Portuguese mono
    tracks = [track(0, "eng", default=True), track(1, "por", channels=1)]
    assert obvious_track(tracks, "en-US") == 0
    # the order doesn't matter
    assert obvious_track(list(reversed(tracks)), "en-US") == 0


def test_two_english_tracks_ask_the_person():
    # A film with a commentary track: both English, so the person decides
    tracks = [track(0, "eng", default=True, channels=6), track(1, "eng", "Director's commentary")]
    assert obvious_track(tracks, "en-US") is None


def test_no_track_in_the_voice_language_asks_too():
    assert obvious_track([track(0, "spa"), track(1, "fra")], "en-US") is None
    assert obvious_track([track(0, "und"), track(1, "und")], "en-US") is None


def test_best_track_for_the_choice_for_all():
    tracks = [track(0, "eng", "Commentary"), track(1, "eng", default=True, channels=6), track(2, "spa")]
    assert best_track_in(tracks, "en") == 1        # not the commentary
    assert best_track_in(tracks, "es") == 2
    assert best_track_in(tracks, "fr") is None     # that file keeps waiting
