import os
import pandas as pd
from pydantic_core import MultiHostUrl
from sqlalchemy import func
from sqlmodel import Field, SQLModel, Session, create_engine, select, delete
from pymilvus import MilvusClient, DataType
from elasticsearch import Elasticsearch

from litellm import completion, embedding
import streamlit as st

from typing import List

from common import LLM_RESPONSE_STYLE, RAG_PROMPT_TEMPLATE

# RDB
engine = create_engine(
    str(
        MultiHostUrl.build(
            scheme="mysql+pymysql",
            username="root",
            password=os.getenv("MYSQL_ROOT_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT", "4000")),
            path=os.getenv("MYSQL_DATABASE")
        )
    )
)

# Embedding
embedding_model = "openai/text-embedding-3-small"
embedding_dimensions = 1536

# LLM
llm_model = "openai/gpt-4o"

# Milvus
milvus_client = MilvusClient(
    uri=f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_GRPC_PORT')}",
    token="root:Milvus"
)
milvus_collection_name = "employee_id_mapping"

# Elasticsearch
es = Elasticsearch(f"http://{os.getenv('ELASTICSEARCH_HOST')}:{os.getenv('ELASTICSEARCH_PORT')}")
ES_INDEX_NAME = "employee_id_mapping"
es_mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "text"},
            "intro": {"type": "text"}
        }
    }
}

if not es.indices.exists(index=ES_INDEX_NAME):
    es.indices.create(index=ES_INDEX_NAME, body=es_mapping)

def get_embedding(text: str) -> List[float]:
    response = embedding(
        model=embedding_model,
        input=[text],
        dimensions=embedding_dimensions
    )
    return response.data[0]['embedding']

# Define and Create the employee table in RDB
class Employee(SQLModel, table=True):
    # adaptor for streamlit
    __table_args__ = {"extend_existing": True}
    id: int | None = Field(default=None, primary_key=True)
    name: str
    intro: str

SQLModel.metadata.create_all(engine)

# Test data
employees = [
    Employee(id=1, name="Cheese Wang", intro="Cheese is a Senior Software Engineer of AI Application at PingCAP, the company behind TiDB"),
    Employee(id=2, name="Hao Huo", intro="Hao Huo is an engineer of PingCAP, the company behind TiDB | Build pingcap/autoflow / OSSInsight.io"),
    Employee(id=3, name="Humble BGL", intro="Humble BGL is the Head of Engagement Innovation at PingCAP, the company behind TiDB")
]

# Define the employee id mapping schema in Milvus
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dimensions)

# Add the employee id mapping index in Milvus
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX"
)

# Create collection of employee id mapping index in Milvus
milvus_client.create_collection(
    collection_name=milvus_collection_name,
    schema=schema,
    index_params=index_params
)

st.title("🔍 Improved RAG Demo")

st.subheader("Database Operations")

left, right = st.columns(2)

if left.button("Reset", type="primary"):
    left.write("Resetting...")
    with Session(engine) as session:
        session.exec(delete(Employee))
        session.commit()

    # Delete the employee id mapping data from Milvus
    milvus_client.delete(
        collection_name=milvus_collection_name,
        ids=[employee.id for employee in employees]
    )

    if es.indices.exists(index=ES_INDEX_NAME):
        es.indices.delete(index=ES_INDEX_NAME)
        es.indices.create(index=ES_INDEX_NAME, body=es_mapping)

    st.rerun()

left.markdown("This option <span style='color: red;'>will delete all data</span></h3>", unsafe_allow_html=True)

if right.button("Save Data"):
    right.write("Saving...")

    # Prepare RDB data
    with Session(engine) as session:
        count = session.exec(select(func.count()).select_from(Employee)).one()

        if count == 0:
            employee_intro_embedding = [
                {
                    "id": employee.id,
                    "embedding": get_embedding(employee.intro)
                } for employee in employees
            ]
            session.add_all(employees)
            session.commit()

            # Insert the employee id mapping data into Milvus
            milvus_client.insert(
                collection_name=milvus_collection_name,
                data=employee_intro_embedding
            )

            for employee in employees:
                es.index(
                    index=ES_INDEX_NAME,
                    id=employee.id,
                    document={"id": employee.id, "name": employee.name, "intro": employee.intro}
                )

            st.rerun()
        else:
            st.error("Data already exists")

with st.expander("📁 All Employees in the MySQL", expanded=False):
    with Session(engine) as session:
        employees = session.exec(select(Employee)).all()
        if employees:
            data = [{'id': employee.id, 'name': employee.name, 'intro': employee.intro} for employee in employees]
            df = pd.DataFrame(data)
            st.dataframe(df)
        else:
            st.info("No data found in MySQL.")

with st.expander("📁 All Employees ID and Embedding Mapping in the Milvus", expanded=False):
    employees = milvus_client.get(
        collection_name=milvus_collection_name,
        ids=[employee.id for employee in employees],
        output_fields=["id", "embedding"]
    )

    if employees:
        data = [{'id': employee['id'], 'embedding': employee['embedding']} for employee in employees]
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.info("No data found in Milvus.")

with st.expander("📁 All Employees in the Elasticsearch", expanded=False):
    response = es.search(index=ES_INDEX_NAME, body={"query": {"match_all": {}}})
    employees = [{"id": item["_id"], "name": item["_source"]["name"], "intro": item["_source"]["intro"]} for item in response["hits"]["hits"]]
    st.dataframe(pd.DataFrame(employees))

st.write(
    "Enter your question, and the system will retrieve relevant knowledge and generate an answer"
)
mode = st.radio("Select Mode:", ["Retrieval Only", "RAG Q&A"])

query = st.text_input("Enter your question:", "")

if st.button("Send") and query:
    with st.spinner("Processing..."):
        # Retrieve relevant chunks in Milvus
        query_embedding = get_embedding(query)
        res = milvus_client.search(
            collection_name=milvus_collection_name,
            anns_field="embedding",
            data=[query_embedding],
            limit=1,
            search_params={"metric_type": "COSINE"},
            output_fields=["id", "embedding"]
        )

        retrieved_milvus = [{"id": item.entity.id, "embedding": item.entity.embedding} for item in res[0]]
        milvus_ids = [item.entity.id for item in res[0]]

        # Retrieve relevant chunks in Elasticsearch
        es_query = {
            "query": {
                "match": {
                    "intro": query
                }
            },
            "size": 1
        }
        res = es.search(index=ES_INDEX_NAME, body=es_query)
        retrieved_es = [{"id": item["_id"], "score": item["_score"]} for item in res["hits"]["hits"]]
        es_ids = [item["_id"] for item in res["hits"]["hits"]]

        # Merge the ids from Milvus and Elasticsearch
        ids = list(set(milvus_ids + es_ids))

        # Retrieve relevant chunks in RDB
        retrieved_mysql = []
        with Session(engine) as session:
            employees = session.exec(select(Employee).where(Employee.id.in_(ids))).all()
            retrieved_mysql = [{'id': employee.id, 'name': employee.name, 'intro': employee.intro} for employee in employees]

        if retrieved_milvus and retrieved_mysql:
            if mode == "Retrieval Only":
                st.write("### Retrieved Embedding and ID in Milvus:")
                st.dataframe(pd.DataFrame(retrieved_milvus))

                st.write("### Retrieved ID and Score in Elasticsearch:")
                st.dataframe(pd.DataFrame(retrieved_es))
                
                st.write("### Retrieved Data in MySQL:")
                st.dataframe(pd.DataFrame(retrieved_mysql))
            else:
                intro = [employee['intro'] for employee in retrieved_mysql]

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

                with st.expander("📚 Retrieved Knowledge", expanded=False):
                    st.write("### Retrieved Embedding and ID in Milvus:")
                    st.dataframe(pd.DataFrame(retrieved_milvus))

                    st.write("### Retrieved ID and Score in Elasticsearch:")
                    st.dataframe(pd.DataFrame(retrieved_es))

                    st.write("### Retrieved Data in MySQL:")
                    st.dataframe(pd.DataFrame(retrieved_mysql))

        else:
            st.info("No relevant information found")