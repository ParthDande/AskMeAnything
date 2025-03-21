from text_embedding import TextEmbedder
import numpy as np 
import faiss 

embedder = TextEmbedder()
embeddings = embedder.embed_text('FY_ResearchPaper.pdf')

embedding_dim = len(embeddings[0])
embeddings_np = np.array(embeddings).astype('float32')

#creating index 
index = faiss.IndexFlatL2(embedding_dim) 
index.add(embeddings_np)

faiss.write_index(index, "my_vector.index")
print(index.ntotal)  # Total number of vectors stored
print(index.d)       # Dimension of each vector



