import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import *
import reflex as rx
import openai


openai_client = openai.OpenAI(
    api_key=OPEN_AI_KEY,
    organization=OPEN_AI_ORG,
    project=OPEN_AI_PROJ
)


class State(rx.State):
    """The app state."""

    _info = ""
    prompt = ""
    llm_output = ""
    processing = False
    complete = False

    def get_answer(self):
        """Get the image from the prompt."""
        if self.prompt == "":
            return rx.window_alert("Prompt Empty")

        self.processing, self.complete = True, False
        yield
        response = openai_client.completions.create(
            prompt=f"""
            That is all the info you know: <info>{self._info}</info>
            How can Karu Labs help a {self.prompt} business?
            If there is no info about that provided, tell that you don't know.
            """, 
            model="gpt-3.5-turbo-instruct",
            max_tokens=680
        )
        self.llm_output = response.choices[0].text
        self.processing, self.complete = False, True


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
app.add_page(index, title="Reflex:DALL-E")