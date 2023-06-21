from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF e Sor")
    st.header("PDF'e sor!")
    
    # pdf yükleme
    pdf = st.file_uploader("PDF i yukle", type ="pdf")


    # metini pdf'den çıkarma
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in  pdf_reader.pages:
            text += page.extract_text()

        # Metini anlamsal bütünlüğü de koruyacak şekilde parçalama
        text_splitter = CharacterTextSplitter(
             separator="\n",
             chunk_size = 1000,
             chunk_overlap = 200,
             length_function = len
        )
        chunks = text_splitter.split_text(text)
        
        #FacebookSearch kullanarak embeddigns oluşturma
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # kullanıcı girişi 
        user_question = st.text_input("PDF hakkında soru sorun..:")
        if user_question:
             docs = knowledge_base.similarity_search(user_question) #FAISS ile soruya benzer parçları getirme

             llm = OpenAI(temperature=0.9)
             chain = load_qa_chain(llm, chain_type="stuff") #Stuff sabit değil! Başka zincir tipleri de var. 
             with get_openai_callback() as cb: #her sorgunun maliyetini terminal ekranına yazmak için oluşturduk.
                response = chain.run(input_documents=docs, question = user_question) #FAISS den çıkan soruya uygun parçları soruyla beraber modele verme
                print(cb)
             st.write(response) 
        
if __name__ == '__main__':
        main()
