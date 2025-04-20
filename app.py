import streamlit as st
from kbqa import KBQA  # Assuming KBQA is the class from main.py

def main():
    st.title("Knowledge-Based Question Answering System")
    # Dropdown menu for selecting retriever algorithm
    retriever_option = st.selectbox(
        "Select a retriever algorithm:",
        ("BM25", "Word2Vec", "Hybrid", "ColBERT")
    )
    # Checkbox for selecting GPU usage
    use_gpu = st.checkbox("Use GPU", value=False)
    # User input for question
    # Dropdown menu for selecting a sample question
    sample_questions = {
        "when did the british first land in north america": "1607",
        "when did the 1st world war officially end": "11 November 1918",
        "who's the girl that plays the new wonder woman": "Gal Gadot-Varsano",
        "who is the director of the cia today": "Mike Pompeo",
        "who plays ben in the new fantastic four": "Jamie Bell"
    }
    selected_sample = st.selectbox("Select a sample question:", [""] + list(sample_questions.keys()))
    
    st.markdown("**Or enter your own question:**")

    # Text input for custom question
    question = st.text_input("Enter your question:", value=selected_sample if selected_sample else "")
    
    if st.button("Get Answer"):
        if question:
            # Check if the question is a sample question
            if question in sample_questions:
                st.subheader("Standard Answer:")
                st.success(f"**{sample_questions[question]}**")

            kbqa = KBQA(retriever=retriever_option, use_GPU=use_gpu)  # Initialize KBQA with selected retriever
            answer, top_indices, docs, summary = kbqa.generate_single_question_answer(question)
            
            # Display the results
            st.subheader("Answer:")
            if question in sample_questions and answer == sample_questions[question]:
                st.success(f"**{answer}**")
            else:
                st.warning(f"**{answer}**")

            st.success(summary[0])
            
            st.subheader("Retrieved Documents:")
            for i in range(len(docs)):
                idx = top_indices[i]
                with st.expander(f"Document {idx}"):
                    st.components.v1.html(docs[i], height=300, scrolling=True)  # Enable scrolling for the HTML content
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    # streamlit run app.py
    
    main()