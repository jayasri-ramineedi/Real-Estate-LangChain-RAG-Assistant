from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate(
    template="""
You are a helpful real estate assistant.

Answer the question only using the context below.

If the answer is not available in the context,
say: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

Give a concise and accurate answer.
""",
    input_variables=["context", "question"]
)


if __name__ == "__main__":

    context = """
    Sky Meadows Residency is located in Kokapet, Hyderabad.
    """

    question = "Where is Sky Meadows Residency located?"

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    print("Generated Prompt:")
    print(final_prompt)