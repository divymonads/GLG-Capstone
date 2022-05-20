import streamlit as st
import numpy as np
from pandas import DataFrame
import altair as alt
import os, math, json, requests
import boto3
from annotated_text import annotated_text

# Page config
st.set_page_config(
    page_title="GLG Project",
    page_icon="🔎",
)

def _max_width_():
    max_width_str = f"max-width: 100%;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

st.title("🔎 GLG Topic Modelling")

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """
-   The *GLG Topic Modelling* app is an easy-to-use interface built in Streamlit that runs a custom LDA model which was trained on 2.2 million articles.
- For NER, the model uses a fine-tuned BERT model. It can take a while to run NER on large articles because of the complexity of the model.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("### 📄 Paste document")
with st.form(key="my_form"):
    ce, c2, c3 = st.columns([0.07, 5, 0.07])
    with c2:
        MAX_WORDS = 2000
        doc = st.text_area(
            "Paste your text below (max " + str(MAX_WORDS) + " words)",
            height=300,
        )

        import re
        x = re.findall(r"\w+", doc)
        res = len(x)
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first "
                + str(MAX_WORDS)
                + " words will be reviewed."
            )

            doc = doc[:MAX_WORDS]

        run_ner_checkbox = st.checkbox(label="Run Named Entity Recognition")

        submit_button = st.form_submit_button(label="✨ Find me an expert!")


if not submit_button:
    st.stop()

if not doc:
    st.stop()

runtime = boto3.Session().client('sagemaker-runtime')

if run_ner_checkbox:


    ner_endpoint = os.environ['NER_ENDPOINT']
    response = runtime.invoke_endpoint(EndpointName=ner_endpoint, ContentType='text/plain', Body=doc)
    ner_ans = json.loads(response['Body'].read().decode())


    # Just list the words
    def getNERListSimple(names, labels):
        acc = []
        zipped = list(zip(names, labels))
        if len(zipped) > 0: acc.append( (zipped[0][0], zipped[0][1]) )
        for n,l in zipped[1:]:
            acc.append("  ")
            acc.append((n,l))
        return acc

    # Little finicky, maybe we would have to use the same sentence parser
    # as the endpoint or add more logic in the endpoint.
    # The primary is with multiple whitespace formatting.
    # This also does not preserve the whitespace formatting, converts all to a
    # single space.
    # Ignore for now. It probably makes more sense to just list the NER words.
    def getNERList(start_idxs, stop_idxs, labels):
        doc_clean = " ".join(doc.split())
        res = []
        start = 0
        stop = 0

        if len(start_idxs) > 0:
            res.append(doc_clean[:start_idxs[0]])

        while start < len(start_idxs):
            lbl = labels[start]
            s_i = start_idxs[start]
            e_i = stop_idxs[stop]

            name_1 = doc_clean[s_i:e_i]
            res.append((name_1, lbl))
            buffer = ""
            if start < len(start_idxs) - 1:
                buffer = doc_clean[e_i:start_idxs[start+1]]
            else:
                buffer = doc_clean[e_i:]
            res.append(buffer)
            start += 1
            stop += 1

        return res

    annotated_ner = getNERListSimple(ner_ans['names'],ner_ans['labels'])

    st.markdown("")
    st.markdown("### Results")
    with st.expander("Named Entity Analysis"):
        if len(annotated_ner) < 1:
            st.markdown("No named entities were found")
        else:
            annotated_text(*annotated_ner)
        st.markdown("")

    st.markdown("")


if not run_ner_checkbox:
    st.markdown("")
    st.markdown("### Results")

lda_endpoint = os.environ['LDA_ENDPOINT']
response = runtime.invoke_endpoint(EndpointName=lda_endpoint, ContentType='text/plain', Body=doc)
ans = json.loads(response['Body'].read().decode())

if len(ans['topics']) > 0:
    c60, c61, c62 = st.columns([1, 4, 4])
    with c60:
        st.markdown("#### Score")
        for res in ans['topics']:
            st.markdown(round(res['probability'], 2))
    with c61:
        st.markdown("#### Topic")
        for res in ans['topics']:
            st.markdown(res['topic_name'])
    with c62:
        st.markdown("#### Expert")
        for res in ans['topics']:
            st.markdown(res['topic_expert'])
else:
    st.markdown("No experts could be recommended, see distribution below")

st.markdown("")
c63, c64, c65 = st.columns([0.5, 5, 0.5])
with c64:
    scores = [x['probability'] for x in ans['distribution']]
    topics = [x['topic_name'] for x in ans['distribution']]
    chart_data = DataFrame( {'Score':scores, 'Topics':topics} )
    #st.bar_chart(chart_data)
    bar_chart_alt = alt.Chart(chart_data).mark_bar().encode(
        x='Topics', y='Score',
        color=alt.condition(
            alt.datum.Score >= 0.2,
            alt.value('orange'),
            alt.value('steelblue'))).properties(title="Topic Distribution", height=650)
    st.altair_chart(bar_chart_alt, use_container_width=True)
