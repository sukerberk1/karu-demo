import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from settings import *
import reflex as rx
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_AI_KEY, temperature=1.5)


def format_docs(docs):
    return "\n\n".join(doc.page_content + "\n===\n" for doc in docs)

class State(rx.State):
    """The app state."""

    _info = ""
    prompt = ""
    llm_output = ""
    processing = False
    complete = False

    def get_answer(self):
        """Get the answer"""
        if self.prompt == "":
            return rx.window_alert("Prompt Empty")
        
        self.processing, self.complete = True, False
        yield
        loader = WebBaseLoader(
        web_paths=("https://www.karulabs.ai/","https://www.karulabs.ai/security", "https://www.karulabs.ai/company"),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("h2-heading-2", 
                        "paragraph-regular-3", 
                        "paragraph-regular-3 text-color-gray-800", 
                        "h2-heading", "paragraph-regular-2", 
                        "uui-text-size-large-5", 
                        "uui-heading-subheading-3", 
                        "paragraph-large-4",
                        "paragraph-regular-3 text-color-gray-800", 
                        "subheading-small-3", 
                        "h1-heading")
                )
            ),
        )
        docs = loader.load()

        # print("Printing all data from the website:")
        # print(format_docs(docs))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever(search_type="similarity")

        retrieved_docs = retriever.invoke(f"Helping a {self.prompt} business")
        
        # print("Printing only selected data:")
        # print(format_docs(retrieved_docs))

        system_prompt = """
            You are an assistant aiming to answer questions about helping a business.
            That is all the info you know: <info>{context}</info>
            This info is in the form of sentence parts that highlight the speciality of Karu Labs company.
            If the info does not fit in the business niche, tell the user that karu cannot help these kind of business and explain why.
            Your answer must be succint and must not exceed 100 words.
            Use markdown format. Make the key words in your answer bold.
            """
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": f"How can Karu Labs help a {self.prompt} business?"})
        self.llm_output = response["answer"]

        self.processing, self.complete = False, True
        vectorstore.delete_collection()


def index():
    return rx.center(
        rx.vstack(
            rx.cond(
                State.complete,
                rx.vstack(
                    rx.markdown(State.llm_output),
                    width="28em",
                ),
            ),
            rx.heading("Karu assistant", font_size="1.5em"),
            rx.flex(
                rx.box("How can Karu Labs help my "),
                rx.input(
                placeholder="[insert domain here] business?",
                on_blur=State.set_prompt,
                width="10em",
                ),
                rx.box("business?"),
                spacing="2",
                align="center"
            ),
            rx.button(
                "Get an answer!", 
                on_click=State.get_answer,
                width="28em",
                loading=State.processing
            ),
            align="center",
        ),
        width="100%",
        height="100vh",
    )

# Add state and page to the app.
app = rx.App()
app.add_page(index, title="Karu assistant")