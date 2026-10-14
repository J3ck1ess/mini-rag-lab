import re


def split_text_recursively(text, chunk_size=300, chunk_overlap=50):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    # 1. Split by paragraphs first
    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # 2. Keep short paragraphs intact
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        # 3. If a paragraph is too long, split it into sentences
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # 4. If a sentence itself is too long, split it by characters
            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                start = 0
                while start < len(sentence):
                    end = min(start + chunk_size, len(sentence))
                    chunks.append(sentence[start:end])
                    start += chunk_size - chunk_overlap

                continue

            # 5. Combine sentences without exceeding chunk_size
            if not current_chunk:
                current_chunk = sentence

            elif len(current_chunk) + 1 + len(sentence) <= chunk_size:
                current_chunk += " " + sentence

            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

    return chunks


# Sample document
document = """
Chapter 1: Annual Leave

Employees receive 15 days of paid annual leave per year.
Employees can apply for annual leave after working for six months.
Leave applications must be submitted at least 3 working days in advance.

Chapter 2: Remote Work

Remote work requires approval from the direct supervisor.
Employees must submit remote work requests through the company portal.
"""


# Generate chunks
chunks = split_text_recursively(
    document,
    chunk_size=100,
    chunk_overlap=20
)


# Display results
print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(f"Length: {len(chunk)} characters")
    print(chunk)