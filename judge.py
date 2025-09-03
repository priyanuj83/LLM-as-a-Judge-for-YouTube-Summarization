from sentence_transformers import SentenceTransformer, util

# Load a free, powerful embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small & fast, great for similarity

def judge_summary(transcript, summaries):
    # Convert summaries from dict to list
    summary_list = list(summaries.values())

    # Get embeddings
    transcript_embedding = model.encode(transcript, convert_to_tensor=True)
    summary_embeddings = model.encode(summary_list, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(transcript_embedding, summary_embeddings)[0]

    # Pick the most similar summary
    best_index = similarities.argmax().item()
    best_summary = summary_list[best_index]

    return best_summary