import os

import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import CharacterTextSplitter


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

load_dotenv()

# Access the environment variables using the os.environ dictionary
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']

# Load, chunk and index the contents of the blog.

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    # web_paths=("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, 
                                               chunk_overlap=500,
                                               separators=["\n\n", 
                                                            "\n", 
                                                            ".", 
                                                            ",", 
                                                            " ", 
                                                            ""],
                                               add_start_index=True)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

Milvus

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", 
                                     search_kwargs={"k": 6},
                                     include_metadata=True)

retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

for doc in retrieved_docs:
    print(f"\nStart index: {doc.metadata['start_index']} with the content:\n{doc.page_content}")

# Perform a similarity search with scores
query = "What are the approaches to Task Decomposition?"
docs_and_scores = vectorstore.similarity_search_with_score(query, k=6)

for doc, score in docs_and_scores:
    print(f"\nStart index: {doc.metadata['start_index']} \
          score: {round(score, 3)} \n{doc.page_content}")


# prompt = hub.pull("rlm/rag-prompt")

# example_messages = prompt.invoke(
#     {"context": "filler context", "question": "filler question"}
#     ).to_messages()

modified_prompt = PromptTemplate(
    input_variables=["max_sentences", "context", "question"],
    template="""
        You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use {max_sentences} sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
)

# example_messages[0].content = example_messages[0].content.replace("three sentences maximum", 
#                                                                   "ten sentences maximum")

# rag_chain = (
#     {"context": retriever | format_docs, 
#      "question": RunnablePassthrough()}
#     | modified_prompt.partial(max_sentences="ten")
#     | llm
#     | StrOutputParser()
# )

# for chunk in rag_chain.stream("What are the approaches to Task Decomposition?"):
#     print(chunk, end="", flush=True)

# vectorstore.as_retriever()
# retriever.get_relevant_documents_with_scores()