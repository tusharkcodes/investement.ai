from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowladgebase.index import store_vector_DB

def split_document_chunks(documents, chunk_size=1000, chunk_overlap=200):
    try:
        # Initialize the text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Attempt to split the documents
        chunks = splitter.split_documents(documents)
        
        # Attempt to store the chunks in the vector database
        store_vector_DB(chunks)
        
        # Return the chunks
        return chunks

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure all dependencies are installed.")
    except AttributeError as e:
        print(f"AttributeError: {e}. Check if the 'documents' object is valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")