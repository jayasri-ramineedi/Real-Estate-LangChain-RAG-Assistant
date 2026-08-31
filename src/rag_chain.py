from langchain_community.vectorstores import FAISS

from embeddings import create_embeddings
from generator import prompt
from evaluation import evaluate_answer

from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()


def create_rag_chain():

    # Load local embedding model
    embeddings = create_embeddings()

    # Load FAISS vector store
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGEL_API_KEY")
    )

    return retriever, llm


if __name__ == "__main__":

    retriever, llm = create_rag_chain()

    # Take question from user
    question = input("Enter your question: ")

    # Retrieve relevant documents
    retrieved_documents = retriever.invoke(question)

    # Combine retrieved documents
    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    # Create final prompt
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Generate answer
    response = llm.invoke(final_prompt)

    # Extract only generated text
    try:
        answer = response.content[0]["text"]
    except (TypeError, IndexError, KeyError):
        answer = response.content

    print("\nANSWER:")
    print(answer)

    # Evaluate generated answer
    evaluation_result = evaluate_answer(
        question=question,
        context=context,
        answer=answer
    )

    print("\n===== RAG EVALUATION =====")
    print(evaluation_result)