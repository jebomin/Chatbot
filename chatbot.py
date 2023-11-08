import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


# 스타일 CSS 코드
st.write(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f37748, #c049c3);
        color: #fff;
        padding: 20px;
    }
    .stApp header {
        background: none;
        margin-bottom: 20px;
    }
    .stTextInput input {
        background: #fff;
        color: #000;
        padding: 10px;
    }
    .stForm button {
        background: #ff4642;
        color: #fff;
        padding: 10px 20px;
        border: none;
    }
    .stForm button:hover {
        background: #ff6460;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('심리상담 챗봇')
st.markdown('본 상담은 어디에도 기록되지 않으니 편하게 상담하셔도 됩니다😀')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
