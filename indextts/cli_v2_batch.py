import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CACHE_PATH = ".indextts2_cli_cache.json"


def _default_session() -> dict[str, Any]:
    return {
        "gpu_pci_uuid": "",
        "model": {
            "cfg_path": "checkpoints/config.yaml",
            "model_dir": "checkpoints",
            "use_fp16": False,
            "use_cuda_kernel": False,
            "use_deepspeed": False,
            "use_accel": False,
            "use_torch_compile": False,
            "device": "",
        },
        "inputs": {
            "voice_reference_path": "examples/voice_01.wav",
            "input_text_file_path": "",
            "output_basename": "gen",
            "emotion_mode": 0,
            "emotion_reference_path": "",
            "emotion_control_weight": 0.65,
            "emotion_text": "",
            "emotion_random": False,
            "emotion_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "audition_text": "欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。",
        },
        "generation": {
            "max_text_tokens_per_segment": 120,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 30,
            "temperature": 0.8,
            "length_penalty": 0.0,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 1500,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_session()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        defaults = _default_session()
        defaults.update(data if isinstance(data, dict) else {})
        for key in ("model", "inputs", "generation"):
            if key not in defaults or not isinstance(defaults[key], dict):
                defaults[key] = _default_session()[key]
            else:
                merged = _default_session()[key].copy()
                merged.update(defaults[key])
                defaults[key] = merged
        return defaults
    except Exception:
        print(f"Warning: failed to read cache file: {path}. Using built-in defaults.")
        return _default_session()


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _prompt_str(label: str, default: str = "", allow_empty: bool = False) -> str:
    while True:
        shown_default = default if default is not None else ""
        value = input(f"{label} [{shown_default}]: ").strip()
        if value == "":
            value = shown_default
        if value == "" and not allow_empty:
            print("Value cannot be empty.")
            continue
        return value


def _prompt_bool(label: str, default: bool) -> bool:
    d = "y" if default else "n"
    while True:
        value = input(f"{label} [y/n, default={d}]: ").strip().lower()
        if value == "":
            return default
        if value in {"y", "yes", "1", "true", "t"}:
            return True
        if value in {"n", "no", "0", "false", "f"}:
            return False
        print("Please enter y or n.")


def _prompt_int(label: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if value == "":
            out = default
        else:
            try:
                out = int(value)
            except ValueError:
                print("Please enter an integer.")
                continue
        if min_value is not None and out < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and out > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return out


def _prompt_float(label: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if value == "":
            out = default
        else:
            try:
                out = float(value)
            except ValueError:
                print("Please enter a number.")
                continue
        if min_value is not None and out < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and out > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return out


def _prompt_existing_path(label: str, default: str, allow_empty: bool = False) -> str:
    while True:
        value = _prompt_str(label, default=default, allow_empty=allow_empty)
        if value == "" and allow_empty:
            return ""
        p = Path(value)
        if p.exists():
            return str(p)
        print(f"Path does not exist: {value}")


def _query_gpu_uuids() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _prompt_gpu_uuid(default: str) -> str:
    env_uuid = os.environ.get("INDEXTTS_GPU_UUID", "").strip()
    if env_uuid:
        print(f"Using GPU UUID from environment INDEXTTS_GPU_UUID={env_uuid}")
        return env_uuid

    uuids = _query_gpu_uuids()
    if uuids:
        print("Detected GPU UUIDs:")
        for idx, uuid in enumerate(uuids):
            print(f"  [{idx}] {uuid}")
    print("Leave empty to disable UUID-based pinning.")

    while True:
        value = _prompt_str("GPU PCI UUID", default=default, allow_empty=True)
        if value == "":
            return ""
        if not uuids or value in uuids:
            return value
        print("UUID not found in current nvidia-smi list. Please enter one from the list, or empty to skip.")


def _pin_gpu_uuid(uuid: str) -> None:
    if not uuid:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["INDEXTTS_GPU_UUID"] = uuid
    print(f"Pinned CUDA_VISIBLE_DEVICES to GPU UUID: {uuid}")


def _collect_inputs(cfg: dict[str, Any], include_model_init: bool = True) -> dict[str, Any]:
    model = cfg["model"]
    inputs = cfg["inputs"]
    gen = cfg["generation"]

    if include_model_init:
        print("\n=== Model initialization ===")
        cfg["gpu_pci_uuid"] = _prompt_gpu_uuid(cfg.get("gpu_pci_uuid", ""))
        model["model_dir"] = _prompt_existing_path("Model directory", model["model_dir"])
        model["cfg_path"] = _prompt_existing_path("Config file path", model["cfg_path"])
        model["use_fp16"] = _prompt_bool("Use FP16", bool(model["use_fp16"]))
        model["use_cuda_kernel"] = _prompt_bool("Use CUDA kernel", bool(model["use_cuda_kernel"]))
        model["use_deepspeed"] = _prompt_bool("Use DeepSpeed", bool(model["use_deepspeed"]))
        model["use_accel"] = _prompt_bool("Use GPT acceleration engine", bool(model["use_accel"]))
        model["use_torch_compile"] = _prompt_bool("Use torch.compile", bool(model["use_torch_compile"]))
        model["device"] = _prompt_str("Device (blank=auto, e.g. cpu/cuda:0/xpu/mps)", model["device"], allow_empty=True)

    print("\n=== Inference inputs ===")
    inputs["voice_reference_path"] = _prompt_existing_path("Voice reference wav path", inputs["voice_reference_path"])
    inputs["input_text_file_path"] = _prompt_existing_path("Main text file path", inputs["input_text_file_path"])
    output_basename = _prompt_str("Output basename (no extension)", inputs["output_basename"])
    inputs["output_basename"] = Path(output_basename).stem

    print("Emotion mode: 0=same-as-speaker, 1=emotion-reference-audio, 2=emotion-vector, 3=emotion-text")
    inputs["emotion_mode"] = _prompt_int("Emotion mode", int(inputs["emotion_mode"]), 0, 3)

    if int(inputs["emotion_mode"]) == 1:
        inputs["emotion_reference_path"] = _prompt_existing_path(
            "Emotion reference wav path",
            inputs["emotion_reference_path"],
        )
    else:
        inputs["emotion_reference_path"] = ""

    inputs["emotion_control_weight"] = _prompt_float(
        "Emotion control weight (emo_alpha)",
        float(inputs["emotion_control_weight"]),
        0.0,
        1.0,
    )

    if int(inputs["emotion_mode"]) == 2:
        vec = inputs.get("emotion_vector", [0.0] * 8)
        new_vec: list[float] = []
        names = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
        for i, name in enumerate(names):
            new_vec.append(_prompt_float(f"Emotion vector [{name}]", float(vec[i]), 0.0, 1.0))
        inputs["emotion_vector"] = new_vec
        inputs["emotion_random"] = _prompt_bool("Emotion random sampling", bool(inputs["emotion_random"]))
        inputs["emotion_text"] = ""
    elif int(inputs["emotion_mode"]) == 3:
        inputs["emotion_text"] = _prompt_str("Emotion description text (empty=use audition/main text)", inputs["emotion_text"], allow_empty=True)
        inputs["emotion_random"] = _prompt_bool("Emotion random sampling", bool(inputs["emotion_random"]))
        inputs["emotion_vector"] = [0.0] * 8
    else:
        inputs["emotion_text"] = ""
        inputs["emotion_random"] = False
        inputs["emotion_vector"] = [0.0] * 8

    inputs["audition_text"] = _prompt_str("Audition text", inputs["audition_text"])

    print("\n=== Advanced generation parameters (Gradio-aligned) ===")
    gen["max_text_tokens_per_segment"] = _prompt_int(
        "max_text_tokens_per_segment",
        int(gen["max_text_tokens_per_segment"]),
        20,
        1000,
    )
    gen["do_sample"] = _prompt_bool("do_sample", bool(gen["do_sample"]))
    gen["top_p"] = _prompt_float("top_p", float(gen["top_p"]), 0.0, 1.0)
    gen["top_k"] = _prompt_int("top_k", int(gen["top_k"]), 0, 100)
    gen["temperature"] = _prompt_float("temperature", float(gen["temperature"]), 0.1, 2.0)
    gen["length_penalty"] = _prompt_float("length_penalty", float(gen["length_penalty"]), -2.0, 2.0)
    gen["num_beams"] = _prompt_int("num_beams", int(gen["num_beams"]), 1, 10)
    gen["repetition_penalty"] = _prompt_float("repetition_penalty", float(gen["repetition_penalty"]), 0.1, 20.0)
    gen["max_mel_tokens"] = _prompt_int("max_mel_tokens", int(gen["max_mel_tokens"]), 50, 10000)

    return cfg


def _read_text(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input text file is empty: {path}")
    return text


def _split_batches(tts: Any, text: str, max_text_tokens_per_segment: int, max_mel_tokens: int) -> list[str]:
    # CLI-side chunking intentionally avoids the known split_text caveat in lower-level generation.
    max_tokens = max(1, min(int(max_text_tokens_per_segment), int(max_mel_tokens)))
    tokens = tts.tokenizer.tokenize(text)
    segments = tts.tokenizer.split_segments(tokens, max_tokens)
    return ["".join(seg).strip() for seg in segments if seg and "".join(seg).strip()]


def _sox_play(path: Path) -> None:
    try:
        subprocess.run(["play", "-q", str(path)], check=True)
    except FileNotFoundError:
        print("SoX player command 'play' was not found. Install SoX to enable audition playback.")
    except subprocess.CalledProcessError as e:
        print(f"Audio playback failed (exit={e.returncode}).")


def _build_generation_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    g = cfg["generation"]
    top_k = int(g["top_k"])
    return {
        "do_sample": bool(g["do_sample"]),
        "top_p": float(g["top_p"]),
        "top_k": top_k if top_k > 0 else None,
        "temperature": float(g["temperature"]),
        "length_penalty": float(g["length_penalty"]),
        "num_beams": int(g["num_beams"]),
        "repetition_penalty": float(g["repetition_penalty"]),
        "max_mel_tokens": int(g["max_mel_tokens"]),
    }


def _infer_once(tts: Any, text: str, out_path: Path, cfg: dict[str, Any]) -> str | None:
    i = cfg["inputs"]
    g = cfg["generation"]

    emo_mode = int(i["emotion_mode"])
    emo_ref_path = None
    emo_vec = None
    use_emo_text = False
    emo_text = None

    if emo_mode == 1:
        emo_ref_path = i["emotion_reference_path"]
    elif emo_mode == 2:
        emo_vec = tts.normalize_emo_vec([float(x) for x in i.get("emotion_vector", [0.0] * 8)], apply_bias=True)
    elif emo_mode == 3:
        use_emo_text = True
        emo_text = i.get("emotion_text", "") or None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    return tts.infer(
        spk_audio_prompt=i["voice_reference_path"],
        text=text,
        output_path=str(out_path),
        emo_audio_prompt=emo_ref_path,
        emo_alpha=float(i["emotion_control_weight"]),
        emo_vector=emo_vec,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=bool(i.get("emotion_random", False)),
        verbose=True,
        max_text_tokens_per_segment=int(g["max_text_tokens_per_segment"]),
        **_build_generation_kwargs(cfg),
    )


def _audition_loop(tts: Any, cfg: dict[str, Any], cache_path: Path) -> bool:
    inputs = cfg["inputs"]
    base = inputs["output_basename"]
    audition_path = Path("outputs") / f"{base}_audition.wav"

    def regenerate_and_play() -> None:
        print("\nGenerating audition sample...")
        _infer_once(tts, inputs["audition_text"], audition_path, cfg)
        _sox_play(audition_path)

    regenerate_and_play()

    while True:
        cmd = input(
            "\nAudition command [Enter=accept, p=play, r=regenerate, m=modify params, q=quit]: "
        ).strip().lower()
        if cmd == "":
            return True
        if cmd == "p":
            _sox_play(audition_path)
            continue
        if cmd == "r":
            regenerate_and_play()
            continue
        if cmd == "m":
            print("\nModify parameters...")
            _collect_inputs(cfg, include_model_init=False)
            _save_json(cache_path, cfg)
            regenerate_and_play()
            continue
        if cmd == "q":
            return False
        print("Unknown command. Use Enter, p, r, m, or q.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IndexTTS2 interactive batch inference with cached defaults and audition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_PATH, help="JSON cache file path")
    args = parser.parse_args()

    cache_path = Path(args.cache_file)
    cfg = _load_json(cache_path)

    try:
        cfg = _collect_inputs(cfg)
        _save_json(cache_path, cfg)

        _pin_gpu_uuid(cfg.get("gpu_pci_uuid", "").strip())

        from indextts.infer_v2 import IndexTTS2

        model = cfg["model"]
        device = model.get("device", "").strip() or None
        tts = IndexTTS2(
            cfg_path=model["cfg_path"],
            model_dir=model["model_dir"],
            use_fp16=bool(model["use_fp16"]),
            device=device,
            use_cuda_kernel=bool(model["use_cuda_kernel"]),
            use_deepspeed=bool(model["use_deepspeed"]),
            use_accel=bool(model["use_accel"]),
            use_torch_compile=bool(model["use_torch_compile"]),
        )

        if not _audition_loop(tts, cfg, cache_path):
            print("Exit requested.")
            return

        main_text = _read_text(cfg["inputs"]["input_text_file_path"])
        chunks = _split_batches(
            tts,
            main_text,
            max_text_tokens_per_segment=int(cfg["generation"]["max_text_tokens_per_segment"]),
            max_mel_tokens=int(cfg["generation"]["max_mel_tokens"]),
        )
        if not chunks:
            raise ValueError("No text chunks generated from input text file.")

        base = cfg["inputs"]["output_basename"]
        out_dir = Path("outputs")
        print(f"\nStarting batch synthesis with {len(chunks)} chunks...")
        for idx, chunk in enumerate(chunks, start=1):
            out_path = out_dir / f"{base}_{idx:03d}.wav"
            print(f"[{idx}/{len(chunks)}] -> {out_path}")
            _infer_once(tts, chunk, out_path, cfg)

        print("Batch synthesis complete.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
