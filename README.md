# High-Performance Fine-Tuning of BERT for Question Answering

This repo contains the code and reports for our HPML final project:

> **Goal:** Accelerate fine-tuning of BERT-base on **SQuAD v2** on a single NVIDIA **T4**, under strict 15–16 GB VRAM and queue constraints, while maintaining near-baseline accuracy.

We explore:
- **System-level optimizations**: SDPA, `torch.compile`, bucketing/pad-max.
- **Parameter-efficient finetuning**: LoRA, BitFit, layer freezing.
- **Memory-oriented tricks**: 8-bit optimizers, gradient checkpointing, AOT-eager.
- **Practical “T4 recipes”**: concrete configs to trade off accuracy, throughput, and VRAM.

---

## 1. Project Milestones and Status

- **Baseline BERT on SQuAD v2 (A1)**  
  - fp16, AdamW, batch size 8, no SDPA, no `torch.compile`.  
  - Establish dev F1, runtime, throughput on a T4.

- **System optimizations (A-series)**  
  - **A2**: +SDPA attention.  
  - **A3**: A2 + `torch.compile` (inductor).  
  - **A4-nb**: A3 + dynamic bucketing / pad-max (no bug).  
  - **A4-b**: buggy bucketing variant that catastrophically breaks accuracy; kept as a negative ablation.

- **Parameter-efficient finetuning (B-series)**  
  - **B1**: LoRA + `torch.compile`.  
  - **B2**: BitFit-style bias-only finetuning.  
  - **B3**: Encoder layer freezing.  
  - Compare speed vs accuracy vs full finetuning.

- **Memory-oriented configs (C-series, D1)**  
  - **C1**: 8-bit optimizer (AdamW 8-bit) + `torch.compile`.  
  - **C2**: Full-precision optimizer + **gradient checkpointing** + `torch.compile`.  
  - **D1**: AOT-eager backend vs inductor.

- **VRAM profiling (A3 vs C1 vs C2)**  
  - `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` on a representative step.  
  - Derive “T4 VRAM recipes” from these numbers.

- **Reports & slides**  
  - `HPML_Project_Proposal.pdf`  
  - `HPML_Midpoint_Report.pdf`  
  - `HPML_Final_Project.pdf` + final presentation slides.

---

## 2. Repository Structure

```text
.
├── README.md                     # This file
├── reports/
│   ├── HPML_Project_Proposal.pdf
│   ├── HPML_Midpoint_Report.pdf
│   └── HPML_Final_Project.pdf
├── scripts/
│   ├── measure_vram_qa.py        # Single-step VRAM measurement (A3, C1, C2)
│   └── aggregate_results.ipynb   # Optional notebook to build plots/tables
└── transformers/                 # (Trimmed) fork of Hugging Face Transformers
    └── examples/pytorch/question-answering/
        ├── run_qa.py             # HF reference script (mostly unchanged)
        ├── trainer_qa.py         # Slightly adapted Trainer for SQuAD v2
        ├── utils_qa.py           # SQuAD v2 preprocessing / postprocessing
        ├── run_qa_a1_baseline.py         # A1: fp16 AdamW, no SDPA, no compile
        ├── run_qa_a2_sdpa.py             # A2: A1 + SDPA
        ├── run_qa_a4_sdpa.py             # A4-nb / A4-b selected via CLI flag
        ├── run_qa_b1_lora.py             # B1: LoRA + compile
        ├── run_qa_b2_bitfit.py           # B2: BitFit-style bias-only
        ├── run_qa_b3_freeze.py           # B3: layer freezing
        ├── run_qa_c1.py                  # C1: 8-bit optimizer + compile
        ├── run_qa_c2.py                  # C2: grad checkpointing + compile
        ├── run_qa_d1.py                  # D1: AOT-eager backend
        └── configs/                      # (optional) JSON/YAML configs per run

If your filenames differ slightly (e.g., run_qa_b3_freeze.py vs run_qa_b3_freeze_layers_t4.py), just update the tree accordingly before committing.


- **Dataset:** [SQuAD v2](https://huggingface.co/datasets/squad_v2) (Stanford Question Answering Dataset v2.0).
- **Access method:** We use the Hugging Face `datasets` library:
  - All training/eval scripts call  
    `load_dataset("squad_v2", trust_remote_code=False, ...)`
    internally via the HF example pipeline.
  - No raw data is committed to this repo; it is downloaded automatically
    on first run and cached under the user’s `~/.cache/huggingface/datasets/`
    directory (or the directory specified via `HF_DATASETS_CACHE`).

- **Task:** Extractive question answering:
  - **Context**: Wikipedia paragraph
  - **Question**: natural language question
  - **Answer**: start/end span in the context (or **no answer** for unanswerable examples)

- **Splits used:**
  - `train` split for fine-tuning
  - `validation` split for evaluation and all reported metrics

- **Preprocessing:**
  - Tokenization via `BertTokenizerFast` with:
    - `max_seq_length = 384`
    - `doc_stride = 128`
    - dynamic overflow handling for long contexts
  - Version with unanswerable questions enabled via
    `--version_2_with_negative`, using the standard SQuAD v2 evaluation
    script (`evaluate.load("squad_v2")`).

To reproduce all results, it is sufficient to:
1. Install `datasets` and `transformers`.
2. Run any of the `run_qa_*.py` scripts with `--dataset_name squad_v2 --version_2_with_negative`.
The dataset will be fetched automatically from the Hugging Face Hub.

3. How to Run the Experiments

All commands assume:
	•	You are inside the transformers repo root on the cluster.
	•	You have a working conda/venv with PyTorch + CUDA and HF dependencies installed.
	•	You run on a single T4 GPU.

⸻

3.1 Environment Setup (example)

# create env
conda create -n hpml_bert python=3.9 -y
conda activate hpml_bert

# install PyTorch + CUDA (version may differ on your cluster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install Transformers and datasets extras
pip install -e .[torch,sentencepiece,vision]
pip install datasets evaluate accelerate bitsandbytes

If you are NOT inside a local editable transformers checkout, you can instead:

pip install transformers[torch,sentencepiece]
pip install datasets evaluate accelerate bitsandbytes

and adjust paths to the run_qa_*.py scripts accordingly.

⸻

3.2 Baseline: A1

cd transformers/examples/pytorch/question-answering

python run_qa_a1_baseline.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --do_train --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --output_dir bert-base-squadv2-baseline-t4 \
  --report_to none


⸻

3.3 System optimizations: A2 / A3 / A4-nb / A4-b

A2: SDPA (no compile)

python run_qa_a2_sdpa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --do_train --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --output_dir bert-base-squadv2-a2-sdpa-t4 \
  --report_to none

A3: A2 + torch.compile (inductor)
(you can either use a dedicated run_qa_a3_sdpa_compile.py or just pass the flags)

python run_qa_a2_sdpa.py \
  --model_name_or_path bert-base-squadv2-a2-sdpa-t4 \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --do_train --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --torch_compile True \
  --torch_compile_backend inductor \
  --output_dir bert-base-squadv2-a3-sdpa-compile-t4 \
  --report_to none

A4-nb vs A4-b: bucketing ablation

# A4-nb: bucketing / pad-max, *no bug*
python run_qa_a4_sdpa.py \
  --bucket_mode nobug \
  ...

# A4-b: buggy variant (used as negative ablation, NOT recommended)
python run_qa_a4_sdpa.py \
  --bucket_mode bug \
  ...

Fill the ... with the same core arguments as A3.

⸻

3.4 Parameter-efficient finetuning: B1 / B2 / B3

B1: LoRA

python run_qa_b1_lora.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --do_train --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --torch_compile True \
  --torch_compile_backend inductor \
  --output_dir bert-base-squadv2-b1-lora-compile-t4 \
  --report_to none

B2: BitFit-style bias-only finetuning

python run_qa_b2_bitfit.py \
  [same core args as above] \
  --output_dir bert-base-squadv2-b2

B3: Freezing encoder layers

python run_qa_b3_freeze.py \
  [same core args as above] \
  --freeze_encoder_layers 10 \
  --output_dir bert-base-squadv2-b3

(Adjust --freeze_encoder_layers to match your script.)

⸻

3.5 Memory-oriented configs: C1 / C2 / D1

C1: 8-bit AdamW optimizer

python run_qa_c1.py \
  [same core args as A3] \
  --optim adamw_bnb_8bit \
  --output_dir bert-base-squadv2-c1-8bit

C2: Full-precision optimizer + gradient checkpointing

python run_qa_c2.py \
  [same core args as A3] \
  --gradient_checkpointing True \
  --output_dir bert-base-squadv2-c2-gc

D1: AOT-eager backend

python run_qa_d1.py \
  [same core args as A3] \
  --torch_compile True \
  --torch_compile_backend aot_eager \
  --output_dir bert-base-squadv2-d1-aot-eager-lr3e-5


⸻

3.6 VRAM measurement (A3 / C1 / C2)

We measure a single training step after a warmup, using torch.cuda.max_memory_allocated() and max_memory_reserved().

cd scripts

python measure_vram_qa.py a3
python measure_vram_qa.py c1
python measure_vram_qa.py c2

This prints peak allocated / reserved VRAM for each config.

⸻

4. Key Results (Short Summary)

Throughput & Accuracy (SQuAD v2 dev):
	•	Baseline A1: ~43.9 samples/s, F1 ≈ 76.6
	•	A3 (SDPA + compile): ~45.3 samples/s, F1 ≈ 76.9
	•	A4-nb (bucketing): ~51.6 samples/s, F1 ≈ 77.2
	•	B1 (LoRA): ~70.4 samples/s, F1 ≈ 54.9
	•	B3 (layer freezing): ~89.8 samples/s, F1 ≈ 72.0
	•	C1 (8-bit opt): ~46.4 samples/s, F1 ≈ 76.3
	•	C2 (GC): ~37.1 samples/s, F1 ≈ 74.5
	•	D1 (AOT-eager): ~45.5 samples/s, F1 ≈ 76.4

VRAM (single-step peak allocated):
	•	A3: ~3087 MiB
	•	C1: ~2466 MiB  (≈ 20% less than A3)
	•	C2: ~2100 MiB  (≈ 32% less than A3)

⸻

5. Observations
	•	System optimizations pay off: SDPA + torch.compile + bucketing (A4-nb) yields ~17% higher throughput vs baseline with slightly better F1.
	•	LoRA + BitFit underperform on SQuAD v2 in our setting: faster, but significantly worse F1 → extractive QA seems to need more capacity / deeper updates.
	•	8-bit optimizers and gradient checkpointing are effective VRAM levers:
	•	C1 keeps accuracy close to A3 while shrinking optimizer state.
	•	C2 gives the strongest VRAM savings with modest F1 drop and lower throughput.
	•	A4-b is a negative ablation: bucketing bug caused catastrophic accuracy collapse (likely label misalignment). We treat it as an autopsy, not a viable config.

⸻

6. How to Reproduce Our Tables and Plots
	1.	Run the desired run_qa_*.py scripts to populate the bert-base-squadv2-* directories.
	2.	Collect training_args.bin and eval_results.json for each run.
	3.	Use scripts/aggregate_results.ipynb to:
	•	Parse metrics across runs.
	•	Generate bar charts (throughput, F1) and VRAM plots.
	4.	Plots used in the final report are saved under reports/figures/ (if you create that folder).

⸻

7. License / Acknowledgements
	•	Built on top of the Hugging Face Transformers￼ library.
	•	SQuAD v2 dataset via Hugging Face Datasets￼.
	•	Project for HPML (High Performance Machine Learning) course.

Once you paste that into `README.md` and adjust any file names that differ on your repo, you’re good for the rubric.
