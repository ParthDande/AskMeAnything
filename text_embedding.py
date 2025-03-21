from text_chunking import PDFChunker 
from sentence_transformers import SentenceTransformer
import pickle



class TextEmbedder:
    def __init__(self,model_name='all-MiniLM-L6-v2'):
        self.chunker = PDFChunker(chunk_size=3)
        self.chunks = None
        self.model = SentenceTransformer(model_name)
        self.embeddings = None 
    def embed_text(self,pdf_path):
        self.chunks = self.chunker.process_pdf(pdf_path)
        self.embeddings = self.model.encode(self.chunks,show_progress_bar=True)
        with open("chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        return self.embeddings
    




