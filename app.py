import streamlit as st
import pandas as pd
import json
import logging
from joblib import load
from cachetools import cached, TTLCache
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_AVAILABLE_MODELS = os.getenv("OLLAMA_AVAILABLE_MODELS", "llama3.1,phi3").split(",")

# Load configuration
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    st.error("Error: config.json file not found. Please ensure it exists in the app directory.")
    logging.error("config.json file not found.")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Error: Invalid JSON in config.json: {str(e)}")
    logging.error(f"Invalid JSON in config.json: {str(e)}")
    st.stop()

# Extract configuration
numerical_cols = [c["name"] for c in config["numerical"]]
categorical_cols = [c["name"] for c in config["categorical"]]
numerical_ranges = {c["name"]: (c["min"], c["max"]) for c in config["numerical"]}
categorical_options = {c["name"]: c["options"] for c in config["categorical"]}

# Load preprocessing objects
try:
    scaler = load(config["scaler_path"])
except FileNotFoundError:
    st.error(f"Error: {config['scaler_path']} file not found. Please ensure it exists.")
    logging.error(f"{config['scaler_path']} file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading {config['scaler_path']}: {str(e)}")
    logging.error(f"Error loading {config['scaler_path']}: {str(e)}")
    st.stop()

# Streamlit app
st.title("Medical Summary Generator")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        [data-testid="stDeployButton"] {
        display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for inputs

# LLM model selection
available_models = OLLAMA_AVAILABLE_MODELS
selected_model = st.sidebar.selectbox("Select LLM Model", available_models, key="llm_model")

st.sidebar.header("Patient Data Input")
patient_data = {}

# Numerical inputs
for col in numerical_cols:
    min_val, max_val = numerical_ranges[col]
    value = st.sidebar.number_input(
        col, min_value=float(min_val), max_value=float(max_val), step=1.0, key=col
    )
    patient_data[col] = value

# Categorical inputs
for col in categorical_cols:
    value = st.sidebar.selectbox(col, options=categorical_options[col], key=col)
    patient_data[col] = value

# Initialize Ollama LLM
try:
    llm = Ollama(model=selected_model, base_url=OLLAMA_BASE_URL)
except Exception as e:
    st.error(f"Error connecting to Ollama API at {OLLAMA_BASE_URL}: {str(e)}")
    logging.error(f"Error connecting to Ollama API: {str(e)}")
    st.stop()

# Define prompt template
prompt = PromptTemplate(
    input_variables=["patient_data"],
    template=config["prompt_template"],
)

# Cache LLM responses (TTL = 1 hour)
cache = TTLCache(maxsize=100, ttl=3600)

@cached(cache)
def generate_summary(patient_data_str):
    try:
        response = llm.invoke(prompt.format(patient_data=patient_data_str))
        return response
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return f"An error occurred: {str(e)}"

# Submit button
if st.sidebar.button("Get Medical Summary"):
    # Validate inputs
    if any(v is None for v in patient_data.values()):
        st.error("Please fill in all fields.")
    else:
        # Preprocess numerical data
        numerical_values = [patient_data[col] for col in numerical_cols]
        numerical_df = pd.DataFrame([numerical_values], columns=numerical_cols)
        try:
            scaled_array = scaler.transform(numerical_df)
        except Exception as e:
            st.error(f"Error scaling data: {str(e)}")
            logging.error(f"Error scaling data: {str(e)}")
            st.stop()

        # Prepare data for LLM
        patient_data_str = "\n".join(f"{k}: {v}" for k, v in patient_data.items())
        logging.info(f"Patient Data:\n{patient_data_str}")

        # Generate summary with timer
        with st.spinner("Generating summary..."):
            start_time = time.time()
            response = generate_summary(patient_data_str)
            end_time = time.time()
            elapsed_time = end_time - start_time

        # Display structured output
        st.markdown(response)
        # Display response time
        st.info(f"Response generated in {elapsed_time:.2f} seconds using {selected_model}.")

else:
    st.write("Enter patient data and click 'Get Medical Summary' to see results.")