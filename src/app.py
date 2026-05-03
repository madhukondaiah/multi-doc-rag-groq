import os
import streamlit as st
from multiplefile_rag import process_document_to_chroma_db, answer_question

working_dir = os.getcwd()

st.title("Multiple File Upload RAG using Groq")

# Upload files
upload_files = st.file_uploader(
    "Upload multiple PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Process button (prevents duplicate processing)
if upload_files:
    if st.button("Process Documents"):
        for file in upload_files:
            file_path = os.path.join(working_dir, file.name)

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            process_document_to_chroma_db(file.name)

        st.success("All documents processed successfully!")

# Ask question
user_question = st.text_input("Ask a question")

if user_question:
    with st.spinner("Thinking..."):
        answer, sources = answer_question(user_question)

    st.write("### Answer:")
    st.write(answer)

    st.write("### Sources:")
    for src in sources:
        st.write(f"- {src}")