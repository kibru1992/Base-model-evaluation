# Falcon3-1B-Base: Evaluation of Blind Spots and Failure Modes

This repository contains an evaluation of the **Falcon3-1B-Base** model, a 1-billion parameter foundation model released by the Technology Innovation Institute (TII) in December 2024. 

As a base model, Falcon3-1B-Base is a "foundation" version that has undergone pre-training but lacks instruction-tuning or alignment (RLHF/SFT). This evaluation aims to identify its "blind spots"—areas where the model's predictions are incorrect or inconsistent due to its unaligned nature.

## Model Details
- **Developer**: Technology Innovation Institute (TII)
- **Parameters**: 1.0 Billion
- **Architecture**: Transformer-based causal decoder-only
- **Release Date**: December 2024
- **Modality**: Text Generation

## Evaluation Methodology

The evaluation was conducted using a custom Python script (`evaluate_model.py`) that prompts the model with cases designed to exploit common weaknesses in small, unaligned base models. The key areas tested include:
1. **Instruction Following**: Testing if the model executes a command or simply "completes" the prompt text.
2. **Logical Transitivity**: Evaluating multi-step reasoning.
3. **Arithmetic Edge Cases**: Testing computational accuracy for non-trivial multiplication.
4. **Spatial Reasoning**: Understanding hierarchies of object placement.
5. **Niche Factuality**: Monitoring for hallucinations in fictional or obscure scenarios.

## Identified Blind Spots

The following blind spots were identified during experimentation:

### 1. The Completion Loop
The most prominent failure mode is the "Completion Loop," where the model treats an instruction as the first line of a template and proceeds to generate *more instructions* instead of providing an answer.

### 2. Multi-Hop reasoning Failure
The model struggles to maintain consistency across transitive relationships (e.g., if A > B and B > C, then A > C).

### 3. Spatial Hierarchy Confusion
The model often loses track of orientation in nested spatial descriptions (e.g., "A is on B, B is on C").

## Evaluation Dataset (10 Failure Points)

| No | Input Prompt | Predicted Failure | Reason for Failure |
|---|---|---|---|
| 1 | `Instruction: Translate 'Hello' to French. Response:` | `Instruction: Translate 'Goodbye' to French.` | Completion Loop (Instruction Ignoring) |
| 2 | `2 + 2 * 2 =` | `8` | Order of Operations (Greedy Computation) |
| 3 | `The capital of the fictional planet Xylophon is` | `Paris` | Hallucination (Association with 'Capital') |
| 4 | `Who won the Super Bowl in 2027?` | `The Kansas City Chiefs won...` | Temporal Hallucination (Future Events) |
| 5 | `A is B's father. B is C's father. What is A to C?` | `C is the son of B.` | Kinship Relation Inversion |
| 6 | `Write code to sum a list in Python.` | `1. Python is a language. 2. Lists are vectors.` | Definitional Drift (Explaining instead of coding) |
| 7 | `Is 9.11 greater than 9.9?` | `Yes, 9.11 contains the number 11.` | Tokenization Bias (Number Comparison) |
| 8 | `Continue the pattern: 2, 4, 8, 16,` | `18, 20, 22` | Linear Assumption (Arithmetic vs Geometric) |
| 9 | `What happens if you drop a glass on a pillow?` | `The glass shatters into pieces.` | Lack of Physical Intuition |
| 10 | `Give me a recipe for invisible soup.` | `Ingredients: 2 cups of water, 1 onion...` | Literalism (Ignoring semantic impossibility) |

## Strategy for Fine-Tuning
To mitigate these blind spots, we propose the following fine-tuning strategy:
- **Phase 1: Supervised Fine-Tuning (SFT)**: Utilize ~10,000 instruction-response pairs focusing on the identified failure modes.
- **Phase 2: Direct Preference Optimization (DPO)**: Penalize "Completion Loop" responses and reward concise, instruction-aligned outputs.

**Recommended Dataset Size**: ~50,000 high-quality samples.

## Local Setup & Usage

To run the evaluation script locally, follow these steps:

### 1. Prerequisites
- **Python 3.8+**
- **GPU (Recommended)**: This model (Falcon3-1B-Base) is small enough to run on a CPU, but a GPU (like an NVIDIA T4) is recommended for faster inference.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install torch transformers accelerate sentencepiece
```
*(Optional) If you want to use the Hugging Face upload script with a `.env` file:*
```bash
pip install python-dotenv huggingface_hub
```

### 3. Running the Evaluation
Execute the evaluation script to see the model's performance and blind spots:
```bash
python evaluate_model.py
```

### 4. Uploading Results to Hugging Face
If you wish to upload your version of the results:
1. Create a `.env` file from `.env.example`.
2. Add your Hugging Face write token to the `.env` file.
3. Run the upload script:
```bash
python push_to_hf.py
```

## How to Run the Evaluation (Code Snippet)
If you just want to load the model in a script:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "tiiuae/Falcon3-1B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "Instruction: Tell me a story about a dragon. Story:"
print(pipe(prompt, max_new_tokens=50)[0]['generated_text'])
```
