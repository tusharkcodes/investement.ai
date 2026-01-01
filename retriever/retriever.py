
from knowladgebase.index import load_vector_DB

# vector_db = load_vector_DB("../investement_knowledgebase_faiss")
vector_db = load_vector_DB("/Users/tusharkungar/ClassRoom/Investement.ai/investement_knowledgebase_faiss")

def get_retriever(vector_db, k: int = 3):
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever

retriever = get_retriever(vector_db)

query = "Is gold a good long-term investment?"

docs = retriever.invoke(query)

for i, doc in enumerate(docs, 1):
    print(f"\nResult ***** {i}")
    print(doc.page_content[:300])
