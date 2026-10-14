def split_into_chunks(text, chunk_size=100, chunk_overlap=20):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)

        # The remaining text fits into one chunk.
        if hard_end == text_length:
            chunks.append(text[start:hard_end])
            break

        window = text[start:hard_end]

        # 1. Prefer sentence-ending punctuation.
        sentence_positions = [
            window.rfind(mark) + 1
            for mark in ".!?"
        ]
        sentence_positions = [
            pos for pos in sentence_positions if pos > 0
        ]

        # 2. Otherwise, prefer weaker punctuation.
        if sentence_positions:
            split_pos = max(sentence_positions)
        else:
            punctuation_positions = [
                window.rfind(mark) + 1
                for mark in ",;:"
            ]
            punctuation_positions = [
                pos for pos in punctuation_positions if pos > 0
            ]

            if punctuation_positions:
                split_pos = max(punctuation_positions)
            else:
                # 3. Fall back to a word boundary.
                space_pos = window.rfind(" ")

                if space_pos > 0:
                    split_pos = space_pos
                else:
                    # 4. Hard split only if no suitable boundary exists.
                    split_pos = len(window)

        absolute_end = start + split_pos
        chunk = text[start:absolute_end].rstrip()

        if chunk:
            chunks.append(chunk)

        if absolute_end >= text_length:
            break

        # 5. Calculate the target overlap start.
        target_start = max(start + 1, absolute_end - chunk_overlap)

        # 6. Move forward to the next word boundary if needed.
        next_start = target_start

        while (
            next_start < absolute_end
            and next_start > 0
            and text[next_start - 1] != " "
            and text[next_start] != " "
        ):
            next_start += 1

        # Skip whitespace.
        while next_start < text_length and text[next_start].isspace():
            next_start += 1

        # Ensure progress.
        if next_start <= start:
            next_start = absolute_end

        start = next_start

    return chunks


# Test document
document = (
    "Employees must submit their annual leave applications through "
    "the company portal at least three working days before the "
    "requested leave date, and every application must be approved "
    "by their direct supervisor. Employees receive 15 days of paid "
    "annual leave per year."
)

chunks = split_into_chunks(
    document,
    chunk_size=100,
    chunk_overlap=20
)

print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(f"Length: {len(chunk)} characters")
    print(chunk)

    assert len(chunk) <= 100, (
        f"Chunk {index} exceeds the size limit!"
    )


# 1. Verify the size constraint
assert all(len(chunk) <= 100 for chunk in chunks)

# 2. Reconstruct the original text approximately
reconstructed = chunks[0]

for chunk in chunks[1:]:
    # Find the overlap between the current result and the next chunk.
    max_overlap = min(len(reconstructed), len(chunk))
    overlap = 0

    for size in range(max_overlap, 0, -1):
        if reconstructed.endswith(chunk[:size]):
            overlap = size
            break

    reconstructed += chunk[overlap:]

# 3. Verify that the original text is preserved
assert reconstructed == document.strip(), (
    "Text was lost or duplicated during chunking."
)

print("\nAll validation checks passed.")