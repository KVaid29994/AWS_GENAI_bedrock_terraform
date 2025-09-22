# frontend.py
import streamlit as st
from chatbot_backend import get_conversation_chain

# Initialize session state
if "chain" not in st.session_state:
    st.session_state.chain = get_conversation_chain()
if "history" not in st.session_state:
    st.session_state.history = []

st.title("💬 Chat with Bedrock + LangChain")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = st.session_state.chain.run(input=user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))

# Display conversation
for role, message in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {message}")
    else:
        st.markdown(f"**🤖 {role}:** {message}")
