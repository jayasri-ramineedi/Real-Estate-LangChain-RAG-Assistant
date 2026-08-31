import streamlit as st

from rag_chain import create_rag_chain
from generator import prompt
from evaluation import evaluate_answer


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Real Estate RAG Assistant",
    page_icon="🏠",
    layout="centered"
)


# ==========================================
# Styling
# ==========================================

st.markdown("""
<style>

h1 {
    text-align: center;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: bold;
}

.answer-box {
    background-color: #ffffff;
    color: #222222;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #dddddd;
    margin-top: 15px;
    font-size: 18px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# Title
# ==========================================

st.title("🏠 Real Estate RAG Assistant")

st.write(
    "Ask questions about your real estate documents."
)

st.divider()


# ==========================================
# Load RAG System
# ==========================================

@st.cache_resource
def load_rag_system():

    retriever, llm = create_rag_chain()

    return retriever, llm


retriever, llm = load_rag_system()


# ==========================================
# Question
# ==========================================

question = st.text_input(
    "🔎 Enter your question:",
    placeholder="Example: Where is Sky Meadows Residency located?"
)


# ==========================================
# Ask Question
# ==========================================

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        # --------------------------------------
        # Retrieve relevant documents
        # --------------------------------------

        retrieved_documents = retriever.invoke(question)


        # --------------------------------------
        # Create context
        # --------------------------------------

        context = "\n\n".join(
            document.page_content
            for document in retrieved_documents
        )


        # --------------------------------------
        # Create prompt
        # --------------------------------------

        final_prompt = prompt.format(
            context=context,
            question=question
        )


        # --------------------------------------
        # Generate answer
        # --------------------------------------

        response = llm.invoke(final_prompt)


        # --------------------------------------
        # Extract answer
        # --------------------------------------

        try:

            answer = response.content[0]["text"]

        except (TypeError, IndexError, KeyError):

            answer = response.content


        # ======================================
        # Answer
        # ======================================

        st.subheader("💡 Answer")

        st.markdown(
            f"""
            <div class="answer-box">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ======================================
        # Evaluation
        # ======================================

        evaluation = evaluate_answer(
            question,
            context,
            answer
        )


        st.subheader("📊 Evaluation Metrics")

        # No box here
        st.write(evaluation)


        # ======================================
        # Retrieved Documents
        # ======================================

        with st.expander("📄 View Retrieved Documents"):

            for i, document in enumerate(
                retrieved_documents,
                start=1
            ):

                st.markdown(f"**Document {i}**")

                st.write(document.page_content)

                st.divider()