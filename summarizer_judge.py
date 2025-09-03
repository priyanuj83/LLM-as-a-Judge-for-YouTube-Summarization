import os
# Disable TensorFlow / Keras completely (force PyTorch-only)
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import pipeline
import torch

# Device config (use GPU if available)
device = 0 if torch.cuda.is_available() else -1

# Define summarization models (all PyTorch-based)
MODELS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-large",
    "pegasus": "google/pegasus-xsum"
}

def load_summarizers():
    """Load all summarizer pipelines (PyTorch only)."""
    summarizers = {}
    for name, model in MODELS.items():
        summarizers[name] = pipeline(
            "summarization",
            model=model,
            tokenizer=model,
            device=device,
            framework="pt"   # force PyTorch backend
        )
    return summarizers


def chunk_text(text, max_words=400):
    """
    Splits text into smaller chunks to fit within model token limits.
    Default ~400 words (~600-700 tokens) is safe for most summarizers.
    """
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def generate_summaries(summarizers, transcript, max_tokens=200):
    """
    Generate candidate summaries from multiple models with safe chunking.
    """
    summaries = {}
    chunks = list(chunk_text(transcript, max_words=400))

    for name, pipe in summarizers.items():
        chunk_summaries = []
        for chunk in chunks:
            result = pipe(
                chunk,
                max_length=max_tokens,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]
            chunk_summaries.append(result)
        summaries[name] = " ".join(chunk_summaries)  # merge chunk summaries
    return summaries