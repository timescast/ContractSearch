from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from datasets import Dataset
import gradio as gr

# --- Load Hugging Face tokenizer ---
hf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# optional stopword list (you can expand this or load from file if needed)
stop_words = {"the", "is", "at", "which", "on", "and", "for", "a", "an", "to"}

def preprocess(text):
    tokens = hf_tokenizer.tokenize(text)
    tokens = [t.lower() for t in tokens if t.lower() not in stop_words]
    return tokens

# --- Load contract dataset ---
df = pd.read_csv("contract_clauses_demo.csv")
contract_ds = Dataset.from_pandas(df)

small_ds = contract_ds.select(range(min(20, len(contract_ds))))
corpus = small_ds["context"]

# --- Prepare BM25 ---
tokenized_corpus = [preprocess(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# --- Bi-encoder ---
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeds = bi_encoder.encode(corpus, convert_to_tensor=True)

# --- Cross-encoder ---
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Search function ---
def search(query, top_k=5):
    # BM25
    q_tokens = preprocess(query)
    bm25_scores = np.array(bm25.get_scores(q_tokens))
    bm25_top = np.argsort(-bm25_scores)[:top_k]

    # Bi-encoder
    q_embed = bi_encoder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_embed, corpus_embeds)[0].cpu().numpy()
    bi_top = np.argsort(-cos_scores)[:top_k]

    # Merge candidates
    candidates = set(bm25_top) | set(bi_top)
    pairs = [(query, corpus[idx]) for idx in candidates]

    # Cross-encoder rerank
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip([c[1] for c in pairs], ce_scores),
                      key=lambda x: x[1], reverse=True)

    return {
        "BM25": [(corpus[i], bm25_scores[i]) for i in bm25_top],
        "Bi-Encoder": [(corpus[i], cos_scores[i]) for i in bi_top],
        "Cross-Encoder": reranked[:top_k]
    }


# Preset queries
preset_queries = [
    "early exit clause",
    "automatic extension of contract",
    "late payment penalty",
    "acts of god clause",
    "which law applies",
    "are products guaranteed"
]

def chat_search(query):
    results = search(query, top_k=3)  # your retrieval pipeline
    # Always return top Cross-Encoder result as chatbot answer
    top_doc, _ = results["Cross-Encoder"][0]
    return results, top_doc

with gr.Blocks() as demo:
    # Header
    gr.Markdown(
        """
        <h1 style='font-size:32px; color:#2c3e50; text-align:center;'>
                 Contract Chatbot
        </h1>
        <p style='text-align:center; font-size:16px; color:gray;'>
            Ask questions about contract clauses. Powered by BM25, Bi-Encoder, and Cross-Encoder.
        </p>
        """
    )

    with gr.Row():
        chatbot = gr.Chatbot(label="Chat Window", height=400)

    # Input + preset dropdown
    query_box = gr.Textbox(placeholder="Type your query here...", label="Enter Query")
    preset = gr.Dropdown(preset_queries, label="Preset Queries", value=None)
    search_btn = gr.Button("Search")

    # Retrieval stage selector
    stage = gr.Dropdown(
        choices=["Cross-Encoder", "Bi-Encoder", "BM25"],
        value="Cross-Encoder",
        label="View Results From"
    )
    stage_output = gr.Textbox(label="Retrieved Results", interactive=False, lines=10)

    state_results = gr.State()  # keep retrieval results across functions

    # Main pipeline
    def combined_input(history, user_query, preset_query):
        if user_query.strip():
            query = user_query
        elif preset_query:
            query = preset_query
        else:
            return history, None, history

        results, final_answer = chat_search(query)
        history = history + [(f"User: {query}", f"Answer: {final_answer}")]
        return history, results, history

    search_btn.click(
        fn=combined_input,
        inputs=[chatbot, query_box, preset],
        outputs=[chatbot, state_results, chatbot]
    )

    # Show retrieval results under dropdown
    def show_stage(results, stage_choice):
        if results is None:
            return ""
        lines = [f"🔎 {stage_choice} results:"]
        for doc, _ in results[stage_choice]:
            lines.append(f"- {doc}")
        return "\n\n".join(lines)

    stage.change(
        fn=show_stage,
        inputs=[state_results, stage],
        outputs=stage_output
    )

demo.launch()
