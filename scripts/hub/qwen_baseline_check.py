"""Step 2 of the revised plan: a real Qwen2-Audio baseline, before any edit.

Questions: does Qwen answer from the audio at all? Do its generated answers match the class-token
scores? Does the input format matter?

Recordings: 10 per family per setup (one per surface when possible), 120 in all, audio.
Input formats (rule fixed before looking at results):
    clip, level 0.1        the 4 s clip as used so far (windows at unit RMS, times 0.1)
    clip, level 0.02 / 0.5 same clip, other levels (natural levels are not kept in the cache:
                           every 0.5 s window was normalized to unit RMS when the cache was built)
    8 windows              each 0.5 s window alone; answer = sum of the 4 family log-probabilities
Controls: silence (zeros, 4 s), white noise (4 s, level 0.1), and the clip of another recording of
a different family (swapped audio).

Two readouts, same prompt as before (qwen_audio_features.PROMPT):
    generated  greedy decoding, max 16 tokens, prompt without answer prefix; valid only if exactly
               one of the 4 family words appears
    scored     after "The surface material is", log-probability of the FULL word " wood",
               " metal", " textile", " polymer" (sum over its tokens), argmax
Also: agreement between the two readouts, and the tokenization of each family word.

Run on the hub:  python scripts/hub/qwen_baseline_check.py
Output: analysis/classification/qwen_audio/baseline_check.csv (one row per recording x format)
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qwen_audio_features import ANSWER_PREFIX, MODEL_NAME, MODEL_REVISION, PROMPT, load_clips
from setup_transfer_probes import score
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

FAMILIES = ["wood", "metal", "textile", "polymer"]
OUT = Path("analysis/classification/qwen_audio")


def main():
    selected = pd.read_csv(OUT / "selected_recordings.csv", dtype={"surface": str})
    np.random.default_rng(0)
    picks = []
    for (_dataset, _family), group in selected.groupby(["dataset", "family"]):
        one_per_surface = group.groupby("surface").sample(1, random_state=0)
        extra = group.drop(one_per_surface.index).sample(max(0, 10 - len(one_per_surface)), random_state=0)
        picks.append(pd.concat([one_per_surface, extra]).head(10))
    table = pd.concat(picks)
    clips = load_clips(table.reset_index(drop=True), "audio")
    table = table.reset_index(drop=True)
    table["surface"] = table.dataset + ":" + table.surface
    print(table.groupby(["dataset", "family"]).size().unstack(), flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    processor.tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"],
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda", quantization_config=quantization
    ).eval()
    tokenizer = processor.tokenizer
    # only the last few positions' logits are ever used: avoids full-vocabulary logits for the whole sequence
    lm_head = next(m for n, m in model.named_modules() if n.endswith("lm_head"))
    lm_head.forward = lambda x: torch.nn.functional.linear(x[:, -4:], lm_head.weight)
    word_tokens = {f: tokenizer.encode(" " + f) for f in FAMILIES}
    for f, ids in word_tokens.items():
        print(f"' {f}' -> {tokenizer.convert_ids_to_tokens(ids)}", flush=True)

    conversation = [
        {"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"}, {"type": "text", "text": PROMPT}]}
    ]
    chat = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    def scored(audio_list):
        return np.concatenate([scored_chunk(audio_list[s : s + 16]) for s in range(0, len(audio_list), 16)])

    @torch.inference_mode()
    def scored_chunk(audio_list):
        """Log-probability of each full family word after the answer prefix (sum over its tokens)."""
        out = np.zeros((len(audio_list), 4))
        for k, f in enumerate(FAMILIES):
            ids = word_tokens[f]
            texts = [chat + ANSWER_PREFIX + tokenizer.decode(ids)] * len(audio_list)
            inputs = processor(text=texts, audio=audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(
                "cuda"
            )
            logits = model(**inputs).logits[:, -len(ids) - 1 : -1].float().log_softmax(-1)
            out[:, k] = (
                torch.gather(
                    logits, 2, torch.as_tensor(ids, device="cuda")[None, :, None].expand(len(audio_list), -1, 1)
                )
                .sum((1, 2))
                .cpu()
                .numpy()
            )
        return out

    @torch.inference_mode()
    def generated(audio_list):
        inputs = processor(
            text=[chat] * len(audio_list), audio=audio_list, sampling_rate=16000, return_tensors="pt", padding=True
        ).to("cuda")
        tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        texts = tokenizer.batch_decode(tokens[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        answers = []
        for text in texts:
            found = [f for f in FAMILIES if re.search(rf"\b{f}", text.lower())]
            answers.append(found[0] if len(found) == 1 else "invalid")
        return answers, texts

    swap_rng = np.random.default_rng(1)
    swapped_index = [
        swap_rng.choice(np.flatnonzero((table.family != table.family[i]).to_numpy())) for i in range(len(table))
    ]
    noise = (np.random.default_rng(2).standard_normal(64000) * 0.1).astype(np.float32)
    formats = {
        "clip, level 0.1": lambda i: [clips[i]],
        "clip, level 0.02": lambda i: [clips[i] * 0.2],
        "clip, level 0.5": lambda i: [clips[i] * 5],
        "8 windows": lambda i: [clips[i][s : s + 8000] for s in range(0, len(clips[i]) - 7999, 8000)],
        "control: silence": lambda i: [np.zeros(64000, np.float32)],
        "control: white noise": lambda i: [noise],
        "control: other family's audio": lambda i: [clips[swapped_index[i]]],
    }
    rows = []
    for name, make in formats.items():
        started = time.time()
        for start in range(0, len(table), 8):
            batch = list(range(start, min(start + 8, len(table))))
            pieces = [make(i) for i in batch]
            flat = [p for ps in pieces for p in ps]
            scores = scored(flat)
            answers, texts = generated(flat) if name != "8 windows" else (["n/a"] * len(flat), [""] * len(flat))
            position = 0
            for i, ps in zip(batch, pieces):
                s = scores[position : position + len(ps)].sum(0)
                rows.append(
                    {
                        "format": name,
                        "recording": i,
                        "dataset": table.dataset[i],
                        "surface": table.surface[i],
                        "family": table.family[i],
                        "scored answer": FAMILIES[int(s.argmax())],
                        "generated answer": answers[position],
                        "generated text": texts[position],
                        **{f"log-prob {f}": s[k] for k, f in enumerate(FAMILIES)},
                    }
                )
                position += len(ps)
        print(f"{name}: {time.time() - started:.0f} s", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "baseline_check.csv", index=False)

    results[~results.format.str.startswith("control")]
    print("\nper format: balanced accuracy (scored / generated), answer shares, invalid rate, agreement")
    for name, group in results.groupby("format", sort=False):
        table_g = group.reset_index(drop=True)
        acc_scored = score(table_g, table_g["scored answer"])["balanced accuracy"]
        valid = table_g["generated answer"] != "invalid"
        acc_generated = (
            score(table_g, table_g["generated answer"])["balanced accuracy"] if name != "8 windows" else np.nan
        )
        agreement = (
            (table_g["scored answer"] == table_g["generated answer"])[valid].mean() if name != "8 windows" else np.nan
        )
        print(
            f"  {name:32s} scored {acc_scored:.2f}  generated {acc_generated:.2f}  "
            f"scored answers {table_g['scored answer'].value_counts().to_dict()}  "
            f"invalid {1 - valid.mean():.2f}  agreement {agreement:.2f}"
        )
    base = results[results.format == "clip, level 0.1"].set_index("recording")
    for control in ["control: silence", "control: white noise", "control: other family's audio"]:
        c = results[results.format == control].set_index("recording")
        same = (c["scored answer"] == base["scored answer"]).mean()
        print(f"  same scored answer as with the real audio, {control}: {same:.2f}")
    print("examples of generated text:", base["generated text"].head(6).tolist())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
