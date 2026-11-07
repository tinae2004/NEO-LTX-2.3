"""Optional F5-TTS backend for Shona dialogue.

Install the upstream F5-TTS stack separately. This module is deliberately lazy:
it does not import or allocate the TTS model until a Shona segment is requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ID = "multilingual-tts/F5-TTS-OpenBible-Shona"


class ShonaTTSUnavailable(RuntimeError):
    """Raised when the optional Shona TTS runtime or model files are unavailable."""


@dataclass(frozen=True)
class ShonaVoice:
    reference_audio: Path
    reference_text: str


class F5ShonaTTS:
    """Lazy, reference-conditioned Shona TTS using the published F5-TTS model.

    The default checkpoint is an Open-Bible read-speech baseline (CC BY-SA 4.0),
    not a guarantee of conversational dialect coverage. Supply a clean 5-10
    second Shona reference recording and its exact transcript.
    """

    def __init__(
        self,
        voice: ShonaVoice,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        device: str = "cuda",
    ) -> None:
        self.voice = voice
        self.repo_id = repo_id
        self.device = device
        self._model = None
        self._vocoder = None
        self._infer_process = None
        self._ref_audio_proc = None
        self._ref_text_proc = None
        self._sample_rate = 24_000

    def load(self) -> None:
        """Load the optional F5-TTS runtime and checkpoint on demand."""
        if self._model is not None:
            return
        if not self.voice.reference_audio.is_file():
            raise ShonaTTSUnavailable(
                f"Shona reference audio not found: {self.voice.reference_audio}"
            )
        if not self.voice.reference_text.strip():
            raise ShonaTTSUnavailable("The Shona reference transcript cannot be empty.")

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from hydra.utils import get_class
            from omegaconf import OmegaConf
            from f5_tts.infer.utils_infer import (
                infer_process,
                load_model,
                load_vocoder,
                preprocess_ref_audio_text,
            )
        except ImportError as exc:
            raise ShonaTTSUnavailable(
                "Install the optional F5-TTS, huggingface_hub, hydra-core and "
                "omegaconf dependencies in the speech environment."
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ShonaTTSUnavailable("CUDA was requested for Shona TTS but is unavailable.")

        try:
            checkpoint = hf_hub_download(self.repo_id, "model_last.pt")
            vocab = hf_hub_download(self.repo_id, "vocab.txt")
            config_path = hf_hub_download(
                self.repo_id, "F5-TTS_OpenBible_Shona.yaml"
            )
            model_cfg = OmegaConf.load(config_path)
            model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
            vocoder = load_vocoder(
                vocoder_name="vocos", is_local=False, device=self.device
            )
            model = load_model(
                model_cls,
                model_cfg.model.arch,
                checkpoint,
                mel_spec_type="vocos",
                vocab_file=vocab,
                use_ema=True,
                device=self.device,
            )
            ref_audio, ref_text = preprocess_ref_audio_text(
                str(self.voice.reference_audio), self.voice.reference_text
            )
        except Exception as exc:
            raise ShonaTTSUnavailable(f"Could not load the Shona F5-TTS model: {exc}") from exc

        self._model = model
        self._vocoder = vocoder
        self._infer_process = infer_process
        self._ref_audio_proc = ref_audio
        self._ref_text_proc = ref_text

    @torch.inference_mode() if False else (lambda fn: fn)
    def synthesize(self, text: str, output_path: str | Path) -> Path:
        """Synthesize one Shona dialogue segment to a 24 kHz WAV file."""
        if not text or not text.strip():
            raise ValueError("Shona dialogue text cannot be empty.")
        self.load()
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ShonaTTSUnavailable("Install soundfile to write Shona WAV output.") from exc

        assert self._infer_process is not None
        wav, sample_rate, _ = self._infer_process(
            self._ref_audio_proc,
            self._ref_text_proc,
            text.strip(),
            self._model,
            self._vocoder,
            mel_spec_type="vocos",
            device=self.device,
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        audio = wav.detach().float().cpu().numpy() if hasattr(wav, "detach") else wav
        sf.write(str(destination), audio, int(sample_rate or self._sample_rate))
        return destination

    def unload(self) -> None:
        """Release TTS weights before loading LTX on a 24 GB GPU."""
        self._model = None
        self._vocoder = None
        self._infer_process = None
        self._ref_audio_proc = None
        self._ref_text_proc = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass
