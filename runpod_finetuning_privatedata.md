# RunPod Fine-Tuning with Private Data: Key Learnings

Here are the key learnings and practical takeaways from my fine‑tuning experiment on RunPod with my private data:

## Private data works without being public

- You can use private Hugging Face datasets or files uploaded directly to RunPod. Provide an HF access token only for gated/private repos, or reference local file paths when my data is in the pod's workspace.

## Data format must match the training pipeline

- Axolotl's standard SFT expects Alpaca-style fields ("instruction", "input", "output"). The chat logs were in messages format [{role, content}], which triggered a KeyError when the script looked for "instruction". Before running "axolotl train config.yaml" I need to change config.yaml to make sure change datasets type to "messages":

datasets:

- path: ./my_messages.jsonl
  type: messages
  dataset_format: custom
  dataset_prompt: messages

## OOM is about tokens x batch x precision

- The primary drivers of GPU memory are sequence_len and micro_batch_size. With sequence_len=4096 and micro_batch_size=16, you hit OOM even on a 140 GB GPU.
- Reliable fixes: reduce micro_batch_size, raise gradient_accumulation_steps, shorten sequence_len, enable gradient_checkpointing, and use bf16/fp16 with 8-bit base weights. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to mitigate fragmentation.

## Role preservation strategies

- For single-turn SFT, user→instruction and assistant→output preserves roles implicitly.
- For multi-turn role fidelity, keep messages arrays and ensure your wrapper/prompt template encodes roles (USER/ASSISTANT/SYSTEM) with masking so only assistant targets are optimized.

## Mac‑only local training is feasible for LoRA

- On Apple Silicon, LoRA/QLoRA of 3B–8B models is practical with MPS. Use smaller sequence lengths, micro batches, gradient checkpointing, and Unsloth/TRL/Axolotl for acceleration. Full fine‑tuning of large models is impractical on Mac; stick to adapters.

## Operational hygiene on RunPod

- Confirm artifacts in your output_dir before stopping the pod. Stopping halts billing and keeps volumes; terminating ends the session but volumes persist if configured.
- Use nvidia-smi to kill stray processes after crashes and restart pods to clear fragmented memory.

## A good baseline Axolotl config for stability

- Start conservatively: micro_batch_size: 2, gradient_accumulation_steps: 8, sequence_len: 2048, gradient_checkpointing: true, load_in_8bit: true, bf16: auto, LoRA r: 8–16, alpha: 16–32, dropout: 0.05. Scale up only after stable runs.

## After the fine tuning model is generated

Merge the fine tuning model with the base model and load.

## References

[RunPod Fine-Tuning Console](https://console.runpod.io/fine-tuning)
