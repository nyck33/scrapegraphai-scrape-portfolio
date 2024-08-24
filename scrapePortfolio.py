import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from scrapegraphai.graphs import SmartScraperGraph

# Function to load environment variables and initialize the model instances
def initialize_models():
    llm_model_instance = AzureChatOpenAI(
        openai_api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
        azure_deployment=st.secrets["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_key=st.secrets["AZURE_OPENAI_API_KEY"]
    )

    embedder_model_instance = AzureOpenAIEmbeddings(
        azure_deployment=st.secrets["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        openai_api_version=st.secrets["AZURE_OPENAI_EMBEDDING_API_VERSION"],
        api_key=st.secrets["AZURE_OPENAI_EMBEDDING_API_KEY"]
    )

    # Supposing model_tokens are 100K
    model_tokens_count = 100000

    graph_config = {

        "llm": {"model_instance": llm_model_instance,
                "model_tokens": model_tokens_count},
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

    # Labeled Text Box for User Input
    st.subheader("Enter the details below:")
    
    # User input for the prompt
    prompt = st.text_input(
        label="Enter your query or prompt:",
        value="",  # Default value is empty, allowing user to type their query
        placeholder="Type your question here..."  # Placeholder text for guidance
    )
    
    # User input for the source URL
    source_url = st.text_input(
        label="Enter the source URL:",
        value="https://nyck33.github.io/2021_portfolio/"
    )
    
    # Button to run the Smart Scraper
    if st.button("Run Smart Scraper"):
        if prompt.strip() == "":  # Check if prompt is empty or only whitespace
            st.error("You forgot to write a question!")  # Display error message in red
        elif source_url.strip() == "":  # Check if source URL is empty or only whitespace
            st.error("You forgot to enter the source URL!")  # Display error message in red
        else:
            # Initialize models
            config = initialize_models()

            # Run the SmartScraperGraph
            result = run_smart_scraper(prompt, source_url, config)

            # Display the result as a formatted JSON dictionary
            st.subheader("Scraped Data:")
            st.json(result)

if __name__ == "__main__":
    main()
