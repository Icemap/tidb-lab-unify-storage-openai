from litellm import completion
import streamlit as st

from common import LLM_RESPONSE_STYLE

llm_model = "openai/gpt-4o"

st.title("🔍 Vanilla AI Q&A Demo")


st.write(
    "Enter your question, and the system will answer it based on the built-in knowledge"
)

query = st.text_input("Enter your question:", "")

if st.button("Send") and query:
    with st.spinner("Processing..."):
        prompt = f"Answer the question: {query}"

        # Call LLM to generate answer
        response = completion(
            model=llm_model,
            messages=[{"content": prompt, "role": "user"}],
        )

        st.markdown(f"### 🤖 {llm_model}")
        st.markdown(LLM_RESPONSE_STYLE, unsafe_allow_html=True)

        # show the response
        st.markdown(
            f'<div class="llm-response">{response.choices[0].message.content}</div>',
            unsafe_allow_html=True,
        )