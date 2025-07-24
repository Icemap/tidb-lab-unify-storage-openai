import os
import pandas as pd

from litellm import completion
import streamlit as st

from typing import Optional, Any
from pytidb import TiDBClient
from pytidb.schema import TableModel, Field
from pytidb.embeddings import EmbeddingFunction
from dotenv import load_dotenv

from common import LLM_RESPONSE_STYLE, RAG_PROMPT_TEMPLATE

load_dotenv()

db = TiDBClient.connect(
    host=os.getenv("SERVERLESS_CLUSTER_HOST"),
    port=int(os.getenv("SERVERLESS_CLUSTER_PORT")),
    username=os.getenv("SERVERLESS_CLUSTER_USERNAME"),
    password=os.getenv("SERVERLESS_CLUSTER_PASSWORD"),
    database=os.getenv("SERVERLESS_CLUSTER_DATABASE_NAME"),
    enable_ssl=True,
)

embedding_model = "openai/text-embedding-3-small"
llm_model = "openai/gpt-4o"

text_embedding_function = EmbeddingFunction(
    embedding_model,
    timeout=60
)

table_name = "employees"
class Employee(TableModel, table=True):
    # adaptor for streamlit
    __tablename__ = table_name
    __table_args__ = {"extend_existing": True}
    id: int | None = Field(default=None, primary_key=True)
    name: str
    intro: str
    intro_vec: Optional[Any] = text_embedding_function.VectorField(
        source_field="intro",
    )

# Test data
employees = [
    Employee(id=1, name="Cheese Wang", intro="Cheese is a Senior Software Engineer of AI Application at PingCAP, the company behind TiDB"),
    Employee(id=2, name="Hao Huo", intro="Hao Huo is an engineer of PingCAP, the company behind TiDB | Build pingcap/autoflow / OSSInsight.io"),
    Employee(id=3, name="Humble BGL", intro="Humble BGL is the Head of Engagement Innovation at PingCAP, the company behind TiDB")
]

table = db.open_table(table_name)
if table is None:
    table = db.create_table(schema=Employee)

if not table.has_fts_index("intro"):
    table.create_fts_index("intro")


st.title("🔍 TiDB RAG Demo")

st.subheader("Database Operations")

left, right = st.columns(2)

if left.button("Reset", type="primary"):
    left.write("Resetting...")
    table.truncate()
    st.rerun()

left.markdown("This option <span style='color: red;'>will delete all data</span></h3>", unsafe_allow_html=True)

if right.button("Save Data"):
    right.write("Saving...")
    # insert sample employees
    if table.rows() == 0:
        table.bulk_insert(employees)
        st.rerun()

with st.expander("📁 All Employees in the Database", expanded=False):
    employees = table.query()
    if employees:
        data = [{'id': employee.id, 'name': employee.name, 'intro': employee.intro, 'intro_vec': employee.intro_vec} for employee in employees]
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.info("No data found in database.")

st.write(
    "Enter your question, and the system will retrieve relevant knowledge and generate an answer"
)
mode = st.radio("Select Mode:", ["Retrieval Only", "RAG Q&A"])

query = st.text_input("Enter your question:", "")

if st.button("Send") and query:
    with st.spinner("Processing..."):
        # Retrieve relevant employees
        results = table.search(query, search_type="hybrid").text_column("intro").limit(1)

        if results:
            if mode == "Retrieval Only":
                st.write("### Hybrid Search Results:")
                st.dataframe(results.to_pandas())
            else:
                intro = [employee.intro for employee in results.to_rows()]

                # Build RAG prompt
                context = "\n".join(intro)
                prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

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

                with st.expander("📚 Hybrid Search Retrieved Knowledge", expanded=False):
                    st.dataframe(results.to_pandas())
        else:
            st.info("No relevant information found")