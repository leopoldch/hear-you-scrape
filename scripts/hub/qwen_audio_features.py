"""Extract Qwen2-Audio representations of each recording, on the hub GPU.

For every selected recording, its 0.5 s windows (unit RMS) are joined into one clip of up to 4 s.
Audio is used as is; acceleration (8 kHz) is resampled to 16 kHz and fed as if it were audio.
The prompt asks for the material family and ends with "The surface material is", so the next
token is the answer. Saved per recording and signal:

- audio encoder: mean over the valid frames, at a few encoder layers;
- language model: hidden state of the last prompt token, and mean over the audio tokens, at a
  few layers;
- the logits of the first token of " wood", " metal", " textile", " polymer" (zero-shot answer).

Language model in 4 bits (the hub GPU is a 10.75 GiB slice). Recordings: all of LMT108, at most 40 per
Cluster surface and 80 per SENS3 surface (seed 0).
Run from the project folder on the hub:  python scripts/hub/qwen_audio_features.py [--quick]
Output: analysis/classification/qwen_audio/features_{signal}.npz and selected_recordings.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

QUICK = "--quick" in sys.argv
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
MODEL_REVISION = "0a095220c30b7b31434169c3086508ef3ea5bf0a"
FAMILIES = ["wood", "metal", "textile", "polymer"]
MAX_PER_SURFACE = {"cluster": 40, "lmt108": None, "sens3": 80}
WAVE_RMS = 0.1
BATCH_SIZE = 8
ENCODER_LAYERS = [8, 16, 24, 32]
LLM_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
PROMPT = (
    "Listen to this recording of a probe sliding on a surface. Which material family is the surface: "
    "wood, metal, textile or polymer? Polymer includes plastics and rubber."
)
ANSWER_PREFIX = "The surface material is"

OUT = Path("analysis/classification/qwen_audio")
OUT.mkdir(parents=True, exist_ok=True)


def select_recordings():
    recordings = pd.read_csv("analysis/signal_processing/recordings.csv", dtype={"surface": str})
    recordings["index_in_waves"] = np.arange(len(recordings))
    recordings["waves_file"] = "waveforms.npz"
    sens3 = pd.read_csv("analysis/classification/sens3_recordings.csv", dtype={"surface": str})
    sens3["dataset"] = "sens3"
    sens3["index_in_waves"] = np.arange(len(sens3))
    sens3["waves_file"] = "sens3_waveforms.npz"
    columns = ["dataset", "surface", "family", "velocity", "index_in_waves", "waves_file"]
    everything = pd.concat([recordings[columns], sens3[columns]], ignore_index=True)

    rng = np.random.default_rng(0)
    selected = []
    for (dataset, _), group in everything.groupby(["dataset", "surface"], sort=True):
        limit = MAX_PER_SURFACE[dataset]
        if limit is not None and len(group) > limit:
            group = group.iloc[np.sort(rng.choice(len(group), limit, replace=False))]
        selected.append(group)
    selected = pd.concat(selected, ignore_index=True)
    if QUICK:
        selected = selected.groupby("dataset").head(16).reset_index(drop=True)
    return selected


def load_clips(selected, signal):
    clips = []
    for waves_file, group in selected.groupby("waves_file", sort=False):
        cache = np.load(f"analysis/classification/{waves_file}")
        windows, recording_of_window = cache["audio" if signal == "audio" else "accel"], cache["recording"]
        first = np.searchsorted(recording_of_window, group.index_in_waves.to_numpy())
        last = np.searchsorted(recording_of_window, group.index_in_waves.to_numpy(), side="right")
        for row_index, start, stop in zip(group.index, first, last):
            clip = windows[start:stop].astype(np.float32).reshape(-1) * WAVE_RMS
            if signal == "accel":
                clip = resample_poly(clip, 2, 1).astype(np.float32)
            clips.append((row_index, clip))
    clips.sort(key=lambda pair: pair[0])
    return [clip for _, clip in clips]


def find_submodule(model, suffix):
    name = next(name for name, _ in model.named_modules() if name.endswith(suffix))
    return model.get_submodule(name)


def main():
    selected = select_recordings()
    selected.to_csv(OUT / "selected_recordings.csv", index=False)
    print(selected.groupby(["dataset", "family"]).size().unstack(), flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    processor.tokenizer.padding_side = "left"
    # The hub gives a 10.75 GiB MIG slice: the language model is loaded in 4 bits (NF4);
    # the audio encoder, projector and output head stay in bf16.
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"],
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda", quantization_config=quantization
    ).eval()
    print(f"model loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    family_token_ids = [processor.tokenizer.encode(" " + family)[0] for family in FAMILIES]
    for family in FAMILIES:
        print(family, "->", processor.tokenizer.tokenize(" " + family), flush=True)
    assert len(set(family_token_ids)) == len(FAMILIES), "family names share a first token"
    audio_token_id = model.config.audio_token_id

    encoder_outputs = {}
    encoder_layers = find_submodule(model, "audio_tower").layers

    def keep_output(layer_number):
        def hook(_module, _inputs, output):
            encoder_outputs[layer_number] = output[0] if isinstance(output, tuple) else output

        return hook

    for layer_number in ENCODER_LAYERS:
        encoder_layers[layer_number - 1].register_forward_hook(keep_output(layer_number))

    conversation = [
        {"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"}, {"type": "text", "text": PROMPT}]}
    ]
    prompt_text = (
        processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) + ANSWER_PREFIX
    )

    for signal in ["audio", "accel"]:
        clips = load_clips(selected, signal)
        arrays = {f"encoder_layer{n}": [] for n in ENCODER_LAYERS}
        arrays.update({f"llm_last_layer{n}": [] for n in LLM_LAYERS})
        arrays.update({f"llm_audio_layer{n}": [] for n in LLM_LAYERS})
        arrays["family_logits"] = []
        started = time.time()
        for batch_start in range(0, len(clips), BATCH_SIZE):
            batch = clips[batch_start : batch_start + BATCH_SIZE]
            inputs = processor(
                text=[prompt_text] * len(batch), audio=batch, sampling_rate=16000, return_tensors="pt", padding=True
            ).to("cuda")
            with torch.inference_mode():
                output = model(**inputs, output_hidden_states=True)

            valid_frames = (inputs["feature_attention_mask"].sum(1) - 1) // 2 + 1
            frame_mask = torch.arange(1500, device="cuda")[None, :] < valid_frames[:, None]
            for n in ENCODER_LAYERS:
                hidden = encoder_outputs[n].float()
                mean = (hidden * frame_mask[..., None]).sum(1) / frame_mask.sum(1, keepdim=True)
                arrays[f"encoder_layer{n}"].append(mean.cpu().half().numpy())

            audio_mask = (inputs["input_ids"] == audio_token_id).float()[..., None]
            for n in LLM_LAYERS:
                hidden = output.hidden_states[n].float()
                arrays[f"llm_last_layer{n}"].append(hidden[:, -1].cpu().half().numpy())
                audio_mean = (hidden * audio_mask).sum(1) / audio_mask.sum(1)
                arrays[f"llm_audio_layer{n}"].append(audio_mean.cpu().half().numpy())
            arrays["family_logits"].append(output.logits[:, -1, family_token_ids].float().cpu().numpy())

            done = batch_start + len(batch)
            if done % (BATCH_SIZE * 20) == 0 or done == len(clips):
                rate = done / (time.time() - started)
                print(
                    f"{signal}: {done}/{len(clips)} recordings, {rate:.1f}/s, "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                    flush=True,
                )

        name = f"features_{signal}{'_quick' if QUICK else ''}.npz"
        np.savez(OUT / name, **{key: np.concatenate(value) for key, value in arrays.items()})
        print("saved", OUT / name, flush=True)

    predicted = np.array(FAMILIES)[
        np.load(OUT / f"features_audio{'_quick' if QUICK else ''}.npz")["family_logits"].argmax(1)
    ]
    print("zero-shot answers (audio):", pd.Series(predicted).value_counts().to_dict(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
