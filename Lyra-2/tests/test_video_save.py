"""Quick smoke test for video saving — verifies codec="libx264" path works on Windows."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import pytest
import torch


def _make_frames(t=16, h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (t, h, w, 3), dtype=np.uint8)


def test_imageio_video_handler_saves_mp4():
    from lyra_2._ext.imaginaire.utils.easy_io.handlers.imageio_video_handler import ImageioVideoHandler

    frames = _make_frames()
    handler = ImageioVideoHandler()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.mp4")
        with open(path, "wb") as f:
            handler.dump_to_fileobj(frames, f, format="mp4", fps=16, quality=5)
        assert os.path.getsize(path) > 1000, "output file is suspiciously small"


def test_save_img_or_video_saves_mp4():
    from lyra_2._ext.imaginaire.visualize.video import save_img_or_video

    frames = torch.from_numpy(_make_frames()).permute(3, 0, 1, 2).float() / 255.0  # C T H W

    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "out")
        save_img_or_video(frames, stem, fps=16)
        mp4 = stem + ".mp4"
        assert os.path.exists(mp4), f"expected {mp4} to exist"
        assert os.path.getsize(mp4) > 1000, "output file is suspiciously small"


def test_save_output_fallback_to_npz(monkeypatch):
    """If video save raises, save_output must fall back to .npz and not crash."""
    import lyra_2._ext.imaginaire.visualize.video as vmod
    from lyra_2._src.inference.lyra2_ar_inference import save_output

    monkeypatch.setattr(vmod, "save_img_or_video", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("ffmpeg broken")))

    frames_tensor = torch.zeros(1, 1, 3, 4, 64, 64)  # n b c t h w

    with tempfile.TemporaryDirectory() as tmp:
        vid_path = os.path.join(tmp, "out.mp4")
        save_output([frames_tensor.squeeze(0)], vid_path)
        stem = os.path.join(tmp, "out")
        assert os.path.exists(stem + ".npz"), "expected npz fallback"


if __name__ == "__main__":
    test_imageio_video_handler_saves_mp4()
    print("test_imageio_video_handler_saves_mp4 PASSED")
    test_save_img_or_video_saves_mp4()
    print("test_save_img_or_video_saves_mp4 PASSED")
    print("Run pytest tests/test_video_save.py for full suite including fallback test")
