import random
import re
import torch
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline, Pipeline
import streamlit as st
from SessionState import _SessionState, _get_session, _get_state
import tweety
import base64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def download_link(object_to_download, download_filename, download_link_text):
    # some strings <-> bytes conversions necessary. Taken from streamlit forums
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model() -> Pipeline:
    return pipeline("text-generation", model="gpt2")


def main():
    state = _get_state()
    st.set_page_config(page_title="National Novel Generation Month", page_icon="📕")
    model = load_model()
    load_page(state, model)
    state.sync()


def filter_content(text):
    text = re.sub(r"""(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*""", ' ', text)
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r'\S+\.com\S+', '', text)  # remove urls
    text = re.sub(r'\@\w+', '', text)  # remove mentions
    text = re.sub(r'\#\w+', '', text)  # remove hashtags
    text = re.sub(r'\W+', ' ', text)
    text = text.strip()
    return text


def load_page(state: _SessionState, model: TextGenerationPipeline):
    st.title("Story Generator")

    state.input = st.text_input(
        "Enter a Twitter Search term",
        value="Artificial Intelligence"
    )

    set_seed(random.randint(0, 100))

    state.slider = st.slider(
        "Minimum Story Length",
        10_000,
        50_000,
        20_000,
        step=10_000
    )

    button_generate = st.button("Generate Story")

    total_str = ""
    total_words = 0

    if button_generate:
        total_words = 0
        try:
            # This only supports max of 1024. So get enough tweets
            # But we dont use all tweets
            inp_split = tweety.get_tweets(state.input, (state.slider // 1048) + 1)
            print((state.slider // 1048) + 1)
            for each in inp_split:
                st.sidebar.markdown(f"{each}\n")
            inp_split = list(map(lambda x: filter_content(x), inp_split))
        except:
            st.error("Sorry! Twitter is having issues.")
            st.stop()
            inp_split = ["CS3600 is such a good class!"]

        i = 0

        st.sidebar.markdown("# Here are your tweets!\n### Now read them while "
                            " GPT2 generates some content...\n")

        progressbar = st.progress(total_words)

        with st.spinner('AI Thinking in Progress. This might take 10 - 15 minutes... 🤔'):
            while state.slider - total_words >= 0:
                outputs = model(
                    inp_split[i],
                    do_sample=True,
                    max_length=1024,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                )
                progressbar.progress(total_words / state.slider)
                print(outputs)
                output_text = outputs[0]["generated_text"]
                total_str += output_text + "\n"
                total_words = len(total_str)
                print(total_words)
                if i == len(inp_split) - 1:
                    i = 0
                else:
                    i += 1

                if total_words >= state.slider:
                    progressbar.empty()

    print("Done")

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
    st.markdown("## Total Words: " + str(len(total_str)))


if __name__ == "__main__":
    main()
