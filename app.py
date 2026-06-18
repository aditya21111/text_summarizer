import os
import tempfile
 
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')

from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,Docx2txtLoader,UnstructuredWordDocumentLoader,UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import chat_models,ChatGoogleGenerativeAI
import streamlit as st

reducer_prompt = """
You are an expert document consolidation system.

You are given structured information extracted from document chunks.

Your task:

1. Merge overlapping information.
2. Remove duplicate statements.
3. Preserve every unique fact.
4. Preserve every formula.
5. Preserve every numerical value.
6. Preserve every experiment result.
7. Preserve every table entry.
8. Preserve every definition.
9. Preserve all section hierarchy.

Rules:

- Never remove information because it seems unimportant.
- Never generalize specific values.
- Never replace numbers with descriptions.
- Never drop formulas.
- Never drop hyperparameters.
- Never drop examples.
- Never invent missing information.

Output:

# Detailed Summary

Organize the information using markdown headings and subheadings.
"""







chunk_summary_generic_prompt = """
You are an expert information extraction system.

Your job is NOT to summarize.

Your job is to convert the document chunk into a structured information record.

Extract and preserve:

- headings
- subheadings
- definitions
- concepts
- facts
- claims
- methodologies
- algorithms
- formulas
- equations
- numerical values
- hyperparameters
- experimental settings
- datasets
- tables
- examples
- conclusions
- limitations
- future work

Rules:

- Preserve information verbatim whenever possible.
- Do not explain.
- Do not simplify.
- Do not compress numerical information.
- Do not merge concepts.
- Do not omit technical details.
- Remove only exact repetition.

Output structured markdown.
"""



chunk_summary_prompt=ChatPromptTemplate.from_messages(
    [
        ('system',chunk_summary_generic_prompt),
        
        ('human','{text}')
    ]
)

summarizer_prompt=ChatPromptTemplate.from_messages([
    ('system',reducer_prompt) ,
    ('human','document : {text}')
      
        ])


st.sidebar.title('Settings')
gemini_api_key=st.sidebar.text_input(label='Your Gemini api key',type='password')

st.title('🦜 Text summarizer using Map reduce technique')
uploaded_file = st.file_uploader("Choose a  file",accept_multiple_files=False,type=['pdf','.txt','.doc','.pptx','.docx'])

if gemini_api_key:



    if uploaded_file is None:
        st.stop()

    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=ext
    ) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if ext=='.pdf':
        loader= PyMuPDFLoader(file_path)

    if ext=='.doc':
        loader=UnstructuredWordDocumentLoader(file_path)
    if ext=='.docx':
        loader=Docx2txtLoader(file_path)
    if ext=='.txt':
        loader=TextLoader(file_path)
    if ext=='.pptx':
        loader=UnstructuredPowerPointLoader(file_path)

    docs=loader.load()
    os.unlink(file_path)
    st.success('Files loaded')
    splitter=RecursiveCharacterTextSplitter(chunk_size=2500,chunk_overlap=200)
    splited=splitter.split_documents(docs)
    st.success('Docs splitted')
    llm=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')
    chunk_summ_chain=chunk_summary_prompt|llm

    import time

    st.write(f'Total chunks to process : {len(splited)}')
    st.write(f'approx time (can vary) : {len(splited)*3} seconds')

    all_chunks = []
    with st.spinner("Processing document chunks..."):    
        status = st.empty()
        for i, chunk in enumerate(splited, start=1):
            try:
                result = chunk_summ_chain.invoke(
                    {"text": chunk.page_content}
                )

                all_chunks.append(result.content[0]['text'])
                print(result.content)

                status.write(f"Chunk {i} done")

            except Exception as e:
                print(f"Chunk {i} failed:", e)
                time.sleep(20)


            
    llm=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',temperature=0.5,streaming=True,    thinking_level='medium')
    reducer_chain=summarizer_prompt|llm
    chunked_sum='\n'.join([chunk for chunk in all_chunks])
    response=reducer_chain.stream({'text':chunked_sum,'language':'hindi'})
    placeholder = st.empty()
    final_response = ""

    for chunk in response:
        print(chunk.content)
        try:
            final_response += chunk.content[0]['text']
            placeholder.markdown(final_response + "▌")
        except Exception as e:
            print(e)

        


else:
    st.error('Enter api or anything as key first to continue')



