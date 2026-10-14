document = """
Chapter 1: Annual Leave

Employees receive 15 days of paid annual leave per year.
Employees can apply for annual leave after working for six months.
Leave applications must be submitted at least 3 working days in advance.
All leave applications require approval from the direct supervisor.

Chapter 2: Remote Work

Remote work requires approval from the direct supervisor.
Employees must submit remote work requests through the company portal.

Chapter 3: Health Insurance

The company provides health insurance to all full-time employees.
Insurance coverage begins after the probation period.
"""


# 1. Configure chunking parameters
chunk_size = 300
chunk_overlap = 50


# 2. Split text into chunks
def split_into_chunks(text, chunk_size, chunk_overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


# 3. Generate chunks
chunks = split_into_chunks(
    document,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)


# 4. Display the chunks
print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(f"Length: {len(chunk)} characters")
    print(chunk)