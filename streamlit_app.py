
import streamlit as st
from streamlit_chat import message
import re
from cc_assistant import agent

# --- Page Config ---
st.set_page_config(
    page_title="CreditAssistant: Credit Card Advisor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Global styles */
body {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0a0a0a;
    color: #e0e0e0;
    padding: 0 !important;
}

/* Sidebar */
.stSidebar {
    background-color: #1c2526;
    padding: 1.5rem;
}
.stSidebar h1 {
    color: #d4a017;
    font-size: 2rem;
    margin-bottom: 1rem;
}
.stSidebar .stMarkdown, .stSidebar label {
    color: #c0c0c0;
    font-size: 1.1rem;
}

/* Main content */
.main {
    padding: 0 2rem;
}
.main .block-container {
    background-color: #252c2d;
    border-radius: 12px;
    padding: 2.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

h1 {
    color: #d4a017;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
}
h2 {
    color: #e0e0e0;
    font-size: 2.2rem;
    margin-top: 1.5rem;
}

/* Chat messages */
.stChatMessage {
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    font-size: 1.2rem;
}
[data-testid="stChatMessage"][data-author="user"] {
    background-color: #2a2f30;
    border: 2px solid #d4a017;
    color: #c0c0c0;
}
[data-testid="stChatMessage"][data-author="assistant"] {
    background-color: #3a3f40;
    border: 2px solid #c0c0c0;
    color: #e0e0e0;
}

/* Card styles */
.card {
    background-color: #3a3f40;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s;
}
.card:hover {
    transform: translateY(-4px);
}
.card-title {
    color: #d4a017;
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.card-content {
    color: #e0e0e0;
    font-size: 1.2rem;
}
.badge {
    background-color: #d4a017;
    color: #0a0a0a;
    padding: 0.4rem 0.9rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    margin-right: 0.6rem;
}

/* Buttons */
.stButton > button {
    background: #d4a017;
    color: #0a0a0a;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    font-size: 1.2rem;
    font-weight: 600;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(212, 160, 23, 0.3);
}

/* Chat input */
.stChatInputContainer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #0a0a0a;
    padding: 1.2rem 12.5%;
    z-index: 1000;
    box-sizing: border-box;
}
.stChatInput > div {
    border: 2px solid #d4a017;
    border-radius: 12px;
    background-color: #2a2f30;
    color: #c0c0c0;
    font-size: 1.3rem;
    height: 120px;
    padding: 0.8rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* Chat area padding to avoid overlap with fixed input */
.stChatMessageContainer {
    padding-bottom: 140px;
}

/* Expander */
.stExpander {
    background-color: #3a3f40;
    border: 1px solid #d4a017;
    border-radius: 8px;
}
.stExpander summary {
    color: #e0e0e0;
    font-size: 1.2rem;
}

/* Responsive */
@media (max-width: 768px) {
    .main {
        padding: 0 1rem;
    }
    .main .block-container {
        padding: 1.5rem;
        margin: 0.5rem;
    }
    .stChatInputContainer {
        padding: 0.8rem 5%;
    }
    h1 {
        font-size: 2.4rem;
    }
    h2 {
        font-size: 1.8rem;
    }
    .card {
        padding: 1.2rem;
    }
    .card-title {
        font-size: 1.4rem;
    }
    .card-content {
        font-size: 1.1rem;
    }
    .badge {
        font-size: 0.9rem;
    }
    .stChatMessage {
        font-size: 1.1rem;
    }
    .stChatInput > div {
        font-size: 1.2rem;
        height: 50px;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1>CreditAssistant</h1>", unsafe_allow_html=True)
    st.markdown("Your premier guide to Indian credit cards.")
    
    st.write("**Key Features:**")
    st.write("- **Comprehensive Data:** Leverages a detailed dataset of Indian credit cards.")
    st.write("- **Advanced AI:** Utilizes LangChain and Retrieval Augmented Generation (RAG) for intelligent access to credit card information.")
    st.write("- **Smart Recommendations:** Personalized credit card suggestions based on your needs.")
    
    st.image("https://www.headforpoints.com/wp-content/uploads/2024/03/American-Express-Amex-Centurion-card-webp.webp")
    st.markdown("### Uncover")
    show_details = st.checkbox("Show Behind-the-Scenes Processing", value=False)
    st.session_state["show_details"] = show_details

# --- Main Content ---
left_col, main_col, right_col = st.columns([1, 6, 1])

with main_col:
    st.markdown("<h1>CreditAssistant: Your Credit Card Advisor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #e0e0e0;'>Ask about the best credit cards, compare options, or explore card details.</p>", unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to CreditAssistant! How can I assist you with your credit card needs today?", "details": None}
        ]

    # Display chat messages
    with st.container():
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="💳" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
                if st.session_state.get("show_details", False) and msg.get("details"):
                    with st.expander("Behind-the-Scenes"):
                        try:
                            st.json(msg["details"])
                        except:
                            st.write(msg["details"])

# Chat input
if prompt := st.chat_input("Ask about credit cards (e.g., 'What are the best travel cards')"):
    with main_col:
        st.session_state.messages.append({"role": "user", "content": prompt, "details": None})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="💳"):
            with st.spinner("Finding the best cards..."):
                try:
                    response = agent.invoke({"input": prompt})
                    output = response.get("output", str(response))
                    details = response.get("intermediate_steps", None)
                    
                    if "Top " in output and "Recommendation" in output:
                        lines = output.split("\n")
                        card_section = False
                        current_card = {}
                        cards = []
                        recommendation = ""
                        
                        for line in lines:
                            if line.startswith("Summary of Top "):
                                st.markdown(f"<h2>{line.strip()}</h2>", unsafe_allow_html=True)
                                card_section = True
                            elif card_section and re.match(r"^\d+\.\s+", line):
                                if current_card:
                                    cards.append(current_card)
                                current_card = {"name": re.sub(r"^\d+\.\s*", "", line.strip())}
                            elif card_section and "- " in line:
                                key, value = map(str.strip, line.split(":", 1))
                                key = key.replace("-", "").strip()
                                current_card[key] = value
                            elif line.startswith("Recommendation:"):
                                if current_card:
                                    cards.append(current_card)
                                card_section = False
                                recommendation = line + "\n" + "\n".join(lines[lines.index(line)+1:])
                                break
                        
                        for card in cards:
                            with st.container():
                                st.markdown(f"""
                                <div class="card">
                                    <h4 class="card-title">{card.get('name', 'Unknown')}</h4>
                                    <p class="card-content">
                                        <span class="badge">{card.get('Score', 'N/A')}</span>
                                        <br>Annual Fee: {card.get('Annual Fee', 'N/A')}
                                        <br>Reward Rate: {card.get('Reward Rate', 'N/A')}
                                        <br>Lounge Access: {card.get('Lounge Access', 'N/A')}
                                        <br>Luxury Perks: {card.get('Luxury Perks', 'N/A')}
                                        <br>Welcome Bonus: {card.get('Welcome Bonus', 'N/A')}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if recommendation:
                            st.markdown(f"<div class='card'><h4 class='card-title'>Recommendation</h4><p class='card-content'>{recommendation}</p></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(output)
                    
                    st.session_state.messages.append({"role": "assistant", "content": output, "details": details})
                    
                    if st.session_state.get("show_details", False) and details:
                        with st.expander("Behind-the-Scenes"):
                            try:
                                st.json(details)
                            except:
                                st.write(details)
                
                except Exception as e:
                    error_message = f"Sorry, something went wrong: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message, "details": None})
