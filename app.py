import random
import re
import urllib
from random import randint
import torch
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline, Pipeline
import streamlit as st
from SessionState import _SessionState, _get_session, _get_state
import tweety

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model() -> Pipeline:
    return pipeline("text-generation", model="gpt2")


def main():
    state = _get_state()
    st.set_page_config(page_title="National Novel Generation Month", page_icon="📕")
    model = load_model()
    load_page(state, model)
    state.sync()


def load_page(state: _SessionState, model: TextGenerationPipeline):
    st.title("Story Generator")

    state.input = st.text_input(
        "Enter a Twitter Search term",
        value="#ArtificialIntelligence"
    )

    set_seed(random.randint(0, 100))

    state.slider = st.slider(
        "Story Length",
        10000,
        50000,
        20000,
        step=1000
    )

    button_generate = st.button("Generate Story")

    total_str = ""

    if button_generate:
        total_words = 0
        try:
            # This only supports max of 1024. So get enough tweets
            # But we dont use all tweets
            inp_split = tweety.get_tweets(state.input, (state.slider // 1048) + 1)
            inp_split = list(map(lambda x: re.sub(
                r"""(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*""",
                ' ', x), inp_split))
            inp_split = list(map(lambda x: re.sub(r'\W+', ' ', x), inp_split))
        except:
            st.error("Sorry! Twitter is having issues. But we will use some madeup tweets instead")
            inp_split = "CS3600 is such a cool class"

        i = 0

        st.sidebar.markdown("# Here are your tweets:\n### Now read them while "
                            " GPT2 generates some content.....\n")
        for each in inp_split:
            st.sidebar.markdown(f"{each}\n")

        with st.spinner('AI Thinking in Progress... 🤔'):
            while state.slider - total_words >= 0:
                outputs = model(
                    inp_split[i],
                    do_sample=True,
                    max_length=1024,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                )
                print(outputs)
                output_text = outputs[0]["generated_text"]
                total_str += output_text + "\n"
                total_words = len(total_str)
                print(total_words)
                if i == len(inp_split) - 1:
                    i = 0
                else:
                    i += 1

    print("Done")

    # st.balloons()

    st.markdown(
        '<h2 style="font-family:Courier;text-align:center;">Your Story</h2>',
        unsafe_allow_html=True,
    )

    print(total_str)

    for i, line in enumerate(total_str.split("\n")):
        st.markdown(
            f'<p style="font-family:Courier;text-align:center;">{line}</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
