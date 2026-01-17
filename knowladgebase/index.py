from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

def store_vector_DB(chunks):
    print("Storing chunks in the vector database...")
    
    try:
        # Initialize the embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create the FAISS vector database
        vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Save the vector database locally
        vector_db.save_local("investement_knowledgebase_faiss")
        print("Vector database saved successfully.")

    except ImportError as e:
        print(f"ImportError: {e}. Ensure that all required libraries are installed.")
    except ValueError as e:
        print(f"ValueError: {e}. Check the input chunks or embeddings configuration.")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. Ensure the save path is valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_vector_DB(db_path: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db