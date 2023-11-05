from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


DATA_PATH = "./data"
DB_FAISS_PATH = "vectorstores/db_faiss"

# create vector database:


def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                   chunk_overlap=20)
    texts = text_splitter.split_documents(documents=documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(texts, embedding=embedding)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()


