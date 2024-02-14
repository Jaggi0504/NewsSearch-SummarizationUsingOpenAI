import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool🔗")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f'Enter URL{i+1}')
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")
file_path='faiss_store_openai.pkl'
main_placeholder=st.empty()
llm=OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.9, max_tokens=500)

if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Text Splitter... Started...🙆‍♂️💫")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs=text_splitter.split_documents(data)
    embeddings=OpenAIEmbeddings()
    vectorstore_openai=FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...🙆‍♂️💫")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query=main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstores=pickle.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstores.as_retriever())
            result=chain({'question':query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result['answer'])
            sources=result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list=sources.split("\n")
                for source in sources_list:
                    st.write(source)