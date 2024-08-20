import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from scrapegraphai.graphs import SmartScraperGraph

# Function to load environment variables and initialize the model instances
def initialize_models():
    load_dotenv()
    
    llm_model_instance = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )

    embedder_model_instance = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    graph_config = {
        "llm": {"model_instance": llm_model_instance},
        "embeddings": {"model_instance": embedder_model_instance}
    }
    
    return graph_config

# Function to create and run the SmartScraperGraph
def run_smart_scraper(prompt, source_url, config):
    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=source_url,
        config=config
    )
    
    result = smart_scraper_graph.run()
    return result

# Streamlit app
def main():
    st.title("Smart Scraper with Azure OpenAI")

    # User input for prompt and source URL
    prompt = st.text_area("Enter the prompt:", "Find some information about what does the company do, the name, and a contact email.")
    source_url = st.text_input("Enter the source URL:", "https://scrapegraphai.com/")
    
    if st.button("Run Smart Scraper"):
        # Initialize models
        config = initialize_models()

        # Run the SmartScraperGraph
        result = run_smart_scraper(prompt, source_url, config)

        # Display the result as a formatted JSON dictionary
        st.subheader("Scraped Data:")
        st.json(result)

if __name__ == "__main__":
    main()
