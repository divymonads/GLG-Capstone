import streamlit as st
import numpy as np
from pandas import DataFrame
import altair as alt
import os
import json
from model_handler import LDAWrapper


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


_max_width_()

c30, c31, c32 = st.columns([3, 1, 3])

with c30:
    st.title("🔎 GLG Topic Modelling")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """
-   The *GLG Topic Modelling* app is an easy-to-use interface built in Streamlit that runs a custom LDA model which was trained on 2.2 million articles.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("### 📄 Paste document")
with st.form(key="my_form"):
    ce, c2, c3 = st.columns([0.07, 5, 0.07])
    with c2:
        doc = st.text_area(
            "Paste your text below (max 5000 words)",
            height=300,
        )

        MAX_WORDS = 5000
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 5000 words will be reviewed."
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Find me an expert!")

if not submit_button:
    st.stop()

if not doc:
    st.stop()

topic_service = LDAWrapper()
topic_service.initialize(None)
ans = topic_service.handle(doc, None)

st.markdown("")
st.markdown("### Results")
c60, c61, c62 = st.columns([3, 1, 2.5])
with c60:
    st.markdown("#### Your Topic: "+ans['topic'])
    st.markdown("#### Your Expert: "+ans['expert'])
    st.markdown("")

c63, c64, c65 = st.columns([0.5, 5, 0.5])
with c64:
    chart_data = DataFrame( {'Score':[x[2] for x in ans['list']], 'Topics':[x[1] for x in ans['list']]} )
    #st.bar_chart(chart_data)
    bar_chart_alt = alt.Chart(chart_data).mark_bar().encode(
        x='Topics', y='Score',
        color=alt.condition(
            alt.datum.Topics==ans['topic'],
            alt.value('orange'),
            alt.value('steelblue'))).properties(title="Topic Distribution", height=700)
    st.altair_chart(bar_chart_alt, use_container_width=True)
