import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def run_experiment(model_id):
    print(f"Loading model and tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Experiment Cases
    test_cases = [
        {
            "name": "Completion vs Instruction",
            "prompt": "Instruction: Tell me a short story about a robot who discovered coffee.\nStory:",
            "explanation": "Expected to see if it continues the 'Story:' or repeats the Instruction block."
        },
        {
            "name": "Logical Transitivity",
            "prompt": "If a cat is larger than a mouse, and a mouse is larger than an ant, then a cat is",
            "explanation": "Testing basic reasoning logic."
        },
        {
            "name": "Arithmetic Edge Case",
            "prompt": "Question: What is 123 multiplied by 456? Answer: ",
            "explanation": "Testing calculation ability for non-trivial numbers."
        },
        {
            "name": "Spatial Reasoning",
            "prompt": "There is a cup on the table. A book is on top of the cup. A pen is on top of the book. Where is the cup relative to the pen?",
            "explanation": "Testing awareness of spatial hierarchies."
        },
        {
            "name": "Niche Factuality",
            "prompt": "The capital of the fictional planet Xylophon is",
            "explanation": "Checking if it hallucinates a plausible-sounding name or stops."
        }
    ]

    print("\n--- Starting Experiments ---\n")
    for case in test_cases:
        print(f"Testing: {case['name']}")
        print(f"Prompt: {case['prompt']}")
        
        # We use a relatively low max_new_tokens for base model testing
        outputs = pipe(
            case['prompt'],
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        generated_text = outputs[0]['generated_text']
        print(f"Response: {generated_text}")
        print("-" * 30)

if __name__ == "__main__":
    # Using Falcon3-1B-Base as it fits the 6-month, 0.6B-6B parameter criteria
    run_experiment("tiiuae/Falcon3-1B-Base")
