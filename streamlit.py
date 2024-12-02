import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv('BACKEND_API_URL')

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Predefined questions
def get_predefined_questions():
    return {
        "What factors contribute to the high Score in the data? Are there any common themes or key drivers of customer satisfaction?": "high",
        "How can we further build on the positive trends identified in the high-rated feedback?": "high",
        "For high, What is the average Score in the data? ": "high",
        "What are the main reasons for low Score in the data? Are there consistent issues mentioned by customers?": "low",
        "How can we address the issues identified in the low-rated feedback to improve customer experience?": "low",
        "For low, What is the average Score in the data? ": "low",
        "Are there specific areas that need improvement or attention that can be deduced based on moderate customer satisfaction?": "medium"
    }

# Preprocessing function
def preprocess_data(file):
    data = pd.read_csv(file)
    data['Combined'] = data.apply(lambda x: f"Comment: {x['Comments']}. Score: {x['Score']}", axis=1)
    low_data = data[data['Score'] < 4].sort_values(by='Score', ascending=True)
    medium_data = data[(data['Score'] >= 4) & (data['Score'] <= 7)].sort_values(by='Score', ascending=True)
    high_data = data[data['Score'] > 7].sort_values(by='Score', ascending=True)
    return data, low_data, medium_data, high_data

# Function to calculate statistics
def calculate_statistics(data_subset):
    return {
        "average_rating": data_subset['Score'].mean() if not data_subset.empty else 0,
        "highest_rating": data_subset['Score'].max() if not data_subset.empty else 0,
        "lowest_rating": data_subset['Score'].min() if not data_subset.empty else 0,
    }

# Function to call LLM
def analyze_with_llm(query, data_subset):
    file_contents = '\n'.join(data_subset['Combined'].tolist())
    stats = calculate_statistics(data_subset)
    file_contents += f"\n\n--- Summary Statistics ---\n"
    file_contents += f"Average Score: {stats['average_rating']:.2f}\n"
    file_contents += f"Highest Score: {stats['highest_rating']}\n"
    file_contents += f"Lowest Score: {stats['lowest_rating']}\n"
    prompt = f"""
    Below is the data containing combined comments and ratings, along with summary statistics:

    {file_contents}

    NOTE : All the calculations have been predefined and you do not need to calculate anything from your own knowledge base. Just respond from this file :{file_contents}.

    You are tasked with analyzing this data to provide meaningful insights, identify trends, and suggest actionable recommendations based on the patterns in the scores and comments.

    Return your answers in JSON format structured as:
    {{
        "question": "answer",
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes data and answers questions."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"}
    )
    response_json = response.choices[0].message.content
    response_data = json.loads(response_json)
    # Extract the relevant answer (assuming the response format is correct)
    return response_data.get("question", "No answer found.")

# Streamlit App
st.title("Customer Feedback Analysis")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
    data, low_data, medium_data, high_data = preprocess_data(uploaded_file)

    # Step 2: Show predefined questions
    st.subheader("Ask a Question")
    questions = get_predefined_questions()

    selected_question = st.selectbox("Select a question", list(questions.keys()), index=None)
    
    if selected_question:
        # Get the category for the selected question
        category = questions[selected_question]
        
        # Show a loading message while processing
        st.info("Analyzing... This may take a few seconds.")

        # Choose the correct data subset based on the category
        if category == "low":
            data_subset = low_data
        elif category == "medium":
            data_subset = medium_data
        else:
            data_subset = high_data

        # Call the backend function to get the LLM response
        llm_response = analyze_with_llm(selected_question, data_subset)

        # Display LLM response
        st.subheader("LLM Response")
        st.write(llm_response)
