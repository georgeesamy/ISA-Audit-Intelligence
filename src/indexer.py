import re
import pdfplumber


def chunk_isa_document(pdf_path: str) -> list[dict]:
    chunks = []
    current_chunk_lines = []
    current_para_id = None
    para_pattern = re.compile(r'^(\d+\.|A\d+\.)\s')

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                stripped = line.strip()
                match = para_pattern.match(stripped)
                if match:
                    if current_chunk_lines and current_para_id:
                        chunks.append({
                            "para_id": current_para_id,
                            "text": " ".join(current_chunk_lines).strip(),
                            "source": pdf_path,
                        })
                    current_para_id = match.group(1)
                    current_chunk_lines = [stripped]
                else:
                    if current_chunk_lines:
                        current_chunk_lines.append(stripped)

    if current_chunk_lines and current_para_id:
        chunks.append({
            "para_id": current_para_id,
            "text": " ".join(current_chunk_lines).strip(),
            "source": pdf_path,
        })
    return chunks


def index_isa_corpus(pdf_paths: list[str], isa_index) -> None:
    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(chunk_isa_document(pdf_path))

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = isa_index._encoder.encode(texts, batch_size=32, show_progress_bar=True)
    isa_index._collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[f"{chunk['source']}_{i}" for i, chunk in enumerate(all_chunks)],
        metadatas=[
            {"source": chunk["source"], "para_id": chunk["para_id"]}
            for chunk in all_chunks
        ],
    )
