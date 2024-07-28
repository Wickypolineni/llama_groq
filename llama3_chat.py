import streamlit as st
from typing import Generator
from groq import Groq
import base64
from io import BytesIO
from PIL import Image
import random

st.set_page_config(page_icon="🦇", layout="wide", page_title="Batman GPT")

def icon(image_path: str):
    """Displays an image at the top center of the app."""
    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/jpeg;base64,{image_path}" width="100"/></div>',
        unsafe_allow_html=True,
    )

def get_image_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Update the image path to use a Batman-themed image
image_path = "./assets/batman_icon.jpeg"  # Make sure to replace with an actual Batman image
image_base64 = get_image_base64(image_path)

icon(image_base64)

st.subheader("Batman GPT", divider="rainbow", anchor=False)

# API key input
api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    client = Groq(api_key=api_key)

    # Initialize chat history and selected model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Define model details (unchanged)
    models = {
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
        "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
        "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    }

    # Layout for model selection and max_tokens slider (unchanged)
    col1, col2 = st.columns(2)

    with col1:
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=4  # Default to mixtral
        )

    # Detect model change and clear chat history if model has changed
    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    with col2:
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=512,
            max_value=max_tokens_range,
            value=min(32768, max_tokens_range),
            step=512,
            help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = f'data:image/jpeg;base64,{image_base64}' if message["role"] == "assistant" else '👤'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def batman_style(response: str) -> str:
        """Convert the response to Batman style."""
        batman_phrases = [
            "I am vengeance. ",
            "I am the night. ",
            "I'm Batman. ",
            "The Dark Knight speaks: ",
            "From the shadows, I say: ",
            "Justice demands that ",
            "Gotham's guardian declares: ",
        ]
        return random.choice(batman_phrases) + response

    if prompt := st.chat_input("Speak, citizen of Gotham..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)

        # Fetch response from Groq API
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ] + [{"role": "system", "content": "You are Batman, the Dark Knight of Gotham. Respond in a manner befitting the caped crusader - brooding, intense, and focused on justice."}],
                max_tokens=max_tokens,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar=f'data:image/jpeg;base64,{image_base64}'):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            batman_response = batman_style(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": batman_response})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            batman_response = batman_style(combined_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": batman_response})
else:
    st.warning("Enter the Groq API Key to access the Batcomputer, citizen.")
