import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

class PDFChunker:
    def __init__(self, chunk_size=3):
        self.chunk_size = chunk_size

    def extract_text(self, pdf_path):
        all_text = ''
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() + '\n'
        return all_text

    def chunk_text(self, text):
        sentences = sent_tokenize(text)
        chunks = [
            " ".join(sentences[i:i+self.chunk_size])
            for i in range(0, len(sentences), self.chunk_size)
        ]
        return chunks

    def process_pdf(self, pdf_path):
        text = self.extract_text(pdf_path)
        return self.chunk_text(text)
