"""Failed steps and bad files are explained in plain words."""

from app.errors import explain_failure, friendly_ffmpeg_error


def test_gpu_out_of_memory_suggests_smaller_batch():
    msg = explain_failure("Training", 1, ["Epoch 3", "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate"])
    assert msg.startswith("Training failed: the GPU ran out of memory") and "batch size" in msg


def test_killed_process_means_ram():
    assert "ran out of memory (RAM)" in explain_failure("Training", 137, ["Epoch 1"])


def test_disk_full_and_network():
    assert "disk is full" in explain_failure("Export", 1, ["OSError: [Errno 28] No space left on device"])
    assert "internet connection" in explain_failure("Download", 1, ["urlopen error [Errno -3] Temporary failure in name resolution"])


def test_otherwise_shows_the_error_line_not_progress():
    msg = explain_failure("The augment step", 2, ["ValueError: bad shape", "100%|##########| 10/10"])
    assert msg == "The augment step failed (exit code 2): ValueError: bad shape"


def test_ffmpeg_messages():
    assert friendly_ffmpeg_error("Output file #0 does not contain any stream") == "This file has no audio track."
    assert "isn't a readable audio or video file" in friendly_ffmpeg_error("movie.mkv: Invalid data found when processing input")


def test_unreadable_file_hides_server_paths():
    msg = friendly_ffmpeg_error("/data/voice/voices/x/freeform/1/source.mp3: Invalid argument")
    assert "isn't a readable audio or video file" in msg and "/data" not in msg
    assert "/data" not in friendly_ffmpeg_error("/data/a/b.wav: Something odd happened")
