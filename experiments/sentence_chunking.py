import re


def split_into_sentences(text):
    """Split text into sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def chunk_sentences(sentences, chunk_size=100, overlap_sentences=1):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be non-negative")

    chunks = []
    current_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # Handle a sentence that exceeds the size limit.
        if len(sentence) > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []

            # Temporary fallback: preserve the long sentence.
            # This branch will be improved later.
            chunks.append(sentence)
            continue

        # Check whether the new sentence fits.
        candidate = " ".join(current_sentences + [sentence])

        if len(candidate) <= chunk_size:
            current_sentences.append(sentence)
            continue

        # Save the current chunk.
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # Try to preserve the last N sentences as overlap.
        if overlap_sentences:
            overlap = current_sentences[-overlap_sentences:]
        else:
            overlap = []

        # Check the size of overlap + new sentence.
        candidate = " ".join(overlap + [sentence])

        if len(candidate) <= chunk_size:
            current_sentences = overlap + [sentence]
        else:
            # Drop overlap if it would exceed the size limit.
            current_sentences = [sentence]

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


document = """
Employees receive 15 days of paid annual leave per year.
Employees can apply for annual leave after working for six months.
Leave applications must be submitted at least 3 working days in advance.
All leave applications require approval from the direct supervisor.
"""

sentences = split_into_sentences(document)

chunks = chunk_sentences(
    sentences,
    chunk_size=100,
    overlap_sentences=1
)

print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(f"Length: {len(chunk)} characters")
    print(chunk)
    print(
        f"\nChunk {index}: "
        f"{len(chunk)} characters, "
        f"within limit: {len(chunk) <= 100}"
    )