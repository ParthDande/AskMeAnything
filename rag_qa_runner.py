import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FAISS index + chunks
index = faiss.read_index("my_vector.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load sentence transformer for encoding query
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load tokenizer + model for generation
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Ask user for query
user_query = input("Ask your question: ")

# Convert to embedding
query_embedding = embedder.encode([user_query]).astype("float32")

# Search top 3 chunks from FAISS
D, I = index.search(query_embedding, k=10)
context = " ".join([chunks[i] for i in I[0]])
print(context)
# Format prompt
prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {user_query}"

# Tokenize and generate answer
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🤖 Answer:", answer)
