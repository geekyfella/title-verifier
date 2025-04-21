import streamlit as st
import requests

st.set_page_config(page_title="Title Verifier", layout="centered")

st.title("📰 Newspaper Title Verification System")
title_input = st.text_input("Enter a new newspaper title for verification:")

if st.button("Verify Title"):
    if title_input.strip() == "":
        st.warning("Please enter a title before verifying.")
    else:
        with st.spinner("Verifying title..."):
            try:
                response = requests.post("http://127.0.0.1:8080/verify-title/", json={"title": title_input})
                result = response.json()
                
                st.subheader("Result:")
                st.markdown(f"**Status:** `{result['status'].upper()}`")
                st.markdown(f"**Verification Probability:** `{result['verification_probability']}`")
                st.markdown(f"**Phonetic Similarity Passed:** {'✅' if result['passed_phonetic'] else '❌'}")
                st.markdown(f"**Semantic Similarity Passed:** {'✅' if result['passed_semantic'] else '❌'}")
                
                if result["same_title_exists"]:
                    st.error("❌ This exact title already exists in the database.")
                if result["combination_detected"]:
                    st.warning(f"⚠️ Title appears to be a combination of: {', '.join(result['combination_titles'])}")
                if result.get("warnings"):
                    st.info("⚠️ Warnings:")
                    for w in result["warnings"]:
                        st.markdown(f"- {w}")

                st.success(result["message"])

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
