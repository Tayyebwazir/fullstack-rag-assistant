def chunk_pages(pages:list[dict],
                chunk_size:int=500,
                overlap:int=100,)->list[dict]:
    if chunk_size <=0:
        raise ValueError("Chunk size must be postive")
    if overlap < 0 or overlap >=chunk_size:
        raise ValueError("Overlap must be >= 0 and < chunk_size")
    chunks=[]
    step = chunk_size-overlap
    for page_data in pages:
        text=page_data["text"]
        page_number = page_data["page"]
        
        for start in range(0,len(text), step):
            content = text[start:start+chunk_size].strip()
            if content:
                chunks.append({
                    "content":content,
                    "page":page_number,})
                if start+chunk_size>=len(text):
                    break
    return chunks