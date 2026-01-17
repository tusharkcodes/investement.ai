import asyncio
from knowladgebase.index import load_vector_DB
from langchain_huggingface import HuggingFaceEmbeddings
from agents.index import test_content, fetch_external_knowledge
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage

ollama_model_client = OllamaChatCompletionClient(model="llama2", base_url="http://localhost:11434")


vector_db = load_vector_DB(
    "/Users/tusharkungar/ClassRoom/Investement.ai/investement_knowledgebase_faiss"
)

def get_retriever(vector_db, k: int = 3):
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

async def main():
    retriever = get_retriever(vector_db)

    query = "Is gold a good long-term investment?"
    docs = retriever.invoke(query)

    ans = await test_content(query, docs[0].page_content)


    # Score > 50 is considered good
    # else Do the web scrapping

    if ans.score > 50:
       response =  await fetch_external_knowledge(query)
    #    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #    external_embedding = embeddings.embed_query(response)
       response = await ollama_model_client.create([UserMessage(content=response, source="user")])
       print("Final Response from Ollama Model:", response.content)


    else:
        await fetch_external_knowledge(query)

    # for i, doc in enumerate(docs, 1):
        # print(f"\nResult ***** {i}")
        # print(doc.page_content[:300])

if __name__ == "__main__":
    asyncio.run(main())
