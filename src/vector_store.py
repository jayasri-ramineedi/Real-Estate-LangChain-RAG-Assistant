from langchain_community.vectorstores import FAISS

from embeddings import create_embeddings
from splitter import split_documents
from loader import load_pdfs


def create_vectorstore(chunks, embeddings):

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


if __name__ == "__main__":

    documents = load_pdfs("data/RealEstate")

    chunks = split_documents(documents)

    print("Total chunks:", len(chunks))

    embeddings = create_embeddings()

    print("Creating FAISS vector store...")

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    vectorstore.save_local("faiss_index")

    print("FAISS vector store created successfully")
    print("Total vectors:", vectorstore.index.ntotal)
    print("FAISS index saved successfully")