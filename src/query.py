from langchain_community.vectorstores import FAISS

from embeddings import create_embeddings


def create_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


if __name__ == "__main__":

    embeddings = create_embeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = create_retriever(vectorstore)

    question = "What is the price of Sky Meadows Residency?"

    retrieved_documents = retriever.invoke(question)

    print("Number of retrieved documents:", len(retrieved_documents))

    for i, document in enumerate(retrieved_documents):

        print(f"\n--- Retrieved Document {i + 1} ---")

        print(document.page_content)