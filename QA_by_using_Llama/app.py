import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constant import CHROMA_SETTINGS

checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        task='text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=50,
        do_smaple=True,
        temperature=0.3,
        top_p=0.95

    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db",
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response = ""
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


def main():
    st.title("Search From PDF 🦜")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI Powered Question Answering From Given PDF+
            """
        )

    question = st.text_input("Enter Your Question")

    if st.button("Search"):
        st.info("Your Question " + question)
        st.info("Your Answer ")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)


if __name__ == "__main__":
    main()
