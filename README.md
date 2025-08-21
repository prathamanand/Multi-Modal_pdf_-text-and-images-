# Multimodal PDF RAG (CLIP + FAISS + Gemini 1.5)

Query **PDFs with both text and images**. This notebook builds a lightweight multimodal RAG pipeline that:
- extracts text *and* embedded images from PDFs (PyMuPDF / `fitz`),
- embeds both modalities using **CLIP** (image & text encoders),
- stores unified embeddings in a **FAISS** vector index,
- retrieves the most relevant chunks (text + images) for a query, and
- sends a multimodal prompt (text context + base64 images) to **Gemini 1.5** for answer generation.

> ⚡ Ideal for visually rich documents like reports, brochures, manuals, and academic PDFs where *images matter* as much as text.

---

## 🔍 Key Features

- **Dual-Modal Ingestion**: Parses PDF pages to extract *text chunks* and *embedded images*.
- **Unified Retrieval**: Uses CLIP to embed both text and images into the *same vector space* → one FAISS index.
- **Colab-Ready Secrets**: Reads `GOOGLE_API_KEY` from Colab’s `userdata` secret store.
- **Gemini 1.5 Answering**: Builds a multimodal prompt (text + images as base64) and asks Gemini for grounded answers.
- **Configurable**: You can adjust chunk sizes, top-k retrieval, and model variants.

---

## 🧱 Architecture (High Level)

```
           PDF (text + images)
                   |
         ┌─────────┴─────────┐
         |                   |
      Text Splitter      Image Extractor
         |                   |
  CLIP Text Encoder     CLIP Image Encoder
         └───────┬───────────┘
                 ▼
            FAISS Index
                 |
          Top‑k Nearest Neighbors
                 |
      Build Multimodal Prompt
      (query + text context + images)
                 |
           Gemini 1.5 (Flash)
                 |
              Answer
```

---

## 📦 Tech Stack

- **Python**, **Jupyter/Colab**
- **PyMuPDF (`fitz`)** for PDF parsing
- **Transformers (HuggingFace)**: `openai/clip-vit-base-patch32`
- **FAISS** via `langchain_community.vectorstores.FAISS`
- **Google Generative AI** (`google-generativeai`) for Gemini 1.5
- **Pillow** for image handling
- **NumPy**, **scikit-learn** for cosine sim/helpers

---

## 🛠️ Setup

> Works great on **Google Colab**. For local runs, ensure PyTorch with CPU/GPU is installed.

### 1) Install dependencies
```bash
pip install "pymupdf<1.25" google-generativeai transformers pillow faiss-cpu langchain-community scikit-learn
```

> Note: In the notebook you may also see `!pip install fitz`; modern versions use `pymupdf`. Use one or the other depending on your environment.

### 2) Configure Gemini API key
- **Colab**: `from google.colab import userdata` then store a secret named `GOOGLE_API_KEY` (Colab: 🔒 *Tools → Secrets*).
- **Local**: set an environment variable:
```bash
export GOOGLE_API_KEY="your_key_here"
```

Inside Python:
```python
import os, google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash-latest")
```

---

## ▶️ How to Use (Notebook Flow)

1. **Open the notebook** `Untitled66 (1).ipynb` in Colab or Jupyter.
2. **Install** the dependencies (first cells).
3. **Set API key** (Colab secrets or env var).
4. **Load your PDF** by setting:
   ```python
   pdf_path = "your_file.pdf"
   ```
5. **Run ingestion**:
   - Extract text chunks (split long pages)
   - Extract embedded images (convert to PIL, base64 for LLM, CLIP embeddings)
6. **Build FAISS index** from all CLIP embeddings (text + images).
7. **Ask questions** with:
   ```python
   answer = multimodal_pdf_rag_pipeline("What does the chart on page 3 show?")
   print(answer)
   ```

---

## 🧠 Core Functions (as used in the notebook)

- `embed_text(text: str) -> np.ndarray`: CLIP text features (L2‑normalized).
- `embed_image(image: Union[str, PIL.Image]) -> np.ndarray`: CLIP image features (L2‑normalized).
- `retrieve_multimodal(query: str, k: int=5) -> List[Document]`: CLIP‑based nearest neighbors from FAISS across text & image docs.
- `create_multimodal_message(query, retrieved_docs) -> dict/list`: Builds a content payload (query + text + base64 images) for the model.
- `multimodal_pdf_rag_pipeline(query: str) -> str`: End‑to‑end: retrieve → construct prompt → call Gemini → return answer.

> The notebook uses `openai/clip-vit-base-patch32` and `langchain_community.vectorstores.FAISS` with **precomputed embeddings** (embedding=None in the vector store) and stores metadata like page numbers and image IDs for traceability.

---

## 💡 Example Queries

- “Summarize page 2 and explain the diagram.”  
- “What visual elements are present in the document?”  
- “Compare the table values between page 1 and page 4.”  
- “Explain the process flow shown in the image.”  

---

## 📁 Project Structure (suggested for a repo)

```
multimodal-pdf-rag/
├── notebooks/
│   └── Untitled66 (1).ipynb
├── data/
│   └── sample.pdf
├── src/
│   ├── ingest.py            # PDF parsing & extraction
│   ├── embeddings.py        # CLIP text/image embedding helpers
│   ├── index.py             # FAISS build/load
│   └── pipeline.py          # retrieve + LLM answer
├── requirements.txt
└── README.md
```

---

## 📉 Limitations & Notes

- **CLIP max text length** ≈ 77 tokens; chunking is important for long pages.
- **Image quality** matters. Embedded raster images are fine; vector graphics may need rasterization.
- **GPU recommended** for faster CLIP embedding but CPU also works for small PDFs.
- **Gemini context limits**: If many images are retrieved, you may need to cap *k* or downscale images.
- This is a **prototype**—for production, consider persistence (save FAISS index), retries, and better chunking/metadata.

---

## 🗺️ Roadmap

- [ ] Move from notebook to modular `src/` package.  
- [ ] Persist FAISS index to disk and add CLI.  
- [ ] Add OCR for scanned PDFs (e.g., Tesseract/EasyOCR).  
- [ ] Add page thumbnails in answers.  
- [ ] Support alternative LLMs (GPT‑4o, Claude 3.5, etc.).  

---

## 📝 License

MIT — feel free to use and adapt. If you build on this, a star ⭐️ would be awesome!

