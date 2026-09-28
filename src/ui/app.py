import streamlit as st
from src.generation.pipeline import RAGPipeline
from src.retrieval.vector_store import DenseRetriever

st.set_page_config(page_title="Enterprise Hybrid RAG", layout="wide")

st.title("Enterprise Hybrid RAG: Search & Verified Answers")
st.markdown("Query company internal documentation with grounded citations and verification.")


@st.cache_resource
def load_pipelines():
    rag = RAGPipeline()
    dense_retriever = DenseRetriever()
    return rag, dense_retriever


rag_pipeline, dense_retriever = load_pipelines()

# Sidebar options
st.sidebar.header("Search Settings")
enable_side_by_side = st.sidebar.checkbox("Compare: Hybrid vs Dense-Only", value=True)
top_k = st.sidebar.slider("Top Candidates Fetched", min_value=5, max_value=20, value=10)
top_n = st.sidebar.slider("Final Context Chunks (Reranked)", min_value=1, max_value=5, value=3)

# Search Bar
query = st.text_input("Ask a question about internal documentation:", value="What is the token expiration period?")

if st.button("Run Search", type="primary") and query:
    with st.spinner("Executing retrieval and generation pipeline..."):
        result = rag_pipeline.query(query, top_k_retrieval=top_k, top_n_rerank=top_n)

    if enable_side_by_side:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hybrid Search (Dense + BM25 + Rerank)")
            st.markdown(f"**Generated Answer:**\n\n{result['answer']}")

            # Confidence metrics display
            conf = result.get("confidence", {})
            st.metric("Composite Confidence", f"{int(conf.get('composite_score', 0) * 100)}%")

            st.write("---")
            st.markdown("#### Retrieved Context Chunks:")
            for s in result.get("sources", []):
                with st.expander(f"[{s['citation_id']}] Source: {s['source']}"):
                    st.write(s["content"])

        with col2:
            st.subheader("Dense-Only Search (Baseline)")
            dense_chunks = dense_retriever.search(query, top_k=top_n)
            st.info("Dense search relies purely on vector cosine similarity without BM25 exact-matching.")
            st.markdown("#### Retrieved Vector Chunks:")
            for idx, c in enumerate(dense_chunks):
                with st.expander(f"Rank #{idx + 1} (Score: {round(c['score'], 3)})"):
                    st.write(c["content"])

    else:
        st.subheader("Answer")
        st.markdown(result["answer"])

        st.markdown("### Confidence Breakdown")
        conf = result.get("confidence", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval Strength", f"{int(conf.get('retrieval_confidence', 0) * 100)}%")
        c2.metric("Citation Coverage", f"{int(conf.get('citation_coverage', 0) * 100)}%")
        c3.metric("Final Confidence", f"{int(conf.get('composite_score', 0) * 100)}%")

        st.markdown("### Citation Verification Audit")
        citations = result.get("citations", [])
        if citations:
            for c in citations:
                status = "Supported" if c["is_supported"] else "Ungrounded"
                st.write(f"- Claim: *\"{c['claim']}\"* → [{c['citation_id']}] **{status}**")
        else:
            st.caption("No explicit citations generated.")

        st.markdown("### Retrieved Sources")
        for s in result.get("sources", []):
            with st.expander(f"[{s['citation_id']}] {s['source']}"):
                st.write(s["content"])