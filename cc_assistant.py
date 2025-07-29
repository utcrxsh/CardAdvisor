
import pandas as pd

from typing import List, Dict
import os
import re
import difflib
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.llms.base import LLM
from typing import Any, List, Optional
import logging

# Load Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please set it in your environment.")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Custom Groq LLM class for LangChain compatibility
class GroqLLM(LLM):
    client: Groq
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            # Check for valid response
            if not response or not hasattr(response, 'choices') or not response.choices:
                logging.error(f"Groq API returned empty or malformed response: {response}")
                return "[LLM Error] The language model did not return a valid response. Please try again later."
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API call failed: {e}")
            return f"[LLM Error] The language model failed to respond: {e}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Load credit card data
try:
    df = pd.read_excel("creditcards.xlsx")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "credit_card_name" not in df.columns:
        raise KeyError("Column 'credit_card_name' not found in the Excel file.")
    if df.empty:
        raise ValueError("The Excel file is empty.")

    # Clean annual_fee column
    def clean_currency(value):
        if isinstance(value, str):
            value = value.replace("₹", "").replace(",", "").strip()
            try:
                return float(value)
            except ValueError:
                return 99999.0
        return float(value) if pd.notna(value) else 99999.0

    if "annual_fee" in df.columns:
        df["annual_fee"] = df["annual_fee"].apply(clean_currency)

    # Create combined content for RAG embedding
    df["combined_content"] = df.apply(
        lambda row: (
            f"Card: {row['credit_card_name']}\n"
            f"Reviews: {row.get('cardmaven_review', '') or ''} {row.get('cardexpert_review', '') or ''}\n"
            f"Tags: {row.get('best_tags', '') or ''}\n"
            f"Annual Fee: {row.get('annual_fee', 'Unknown')}"
        ).strip(),
        axis=1
    )
except FileNotFoundError:
    raise FileNotFoundError("The file 'creditcards.xlsx' was not found.")
except Exception as e:
    raise Exception(f"Error loading Excel file: {e}")

# Set up embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index_dir = "./faiss_index"
if not os.path.exists(faiss_index_dir):
    loader = DataFrameLoader(df, page_content_column="combined_content")
    docs = loader.load()
    if not docs:
        raise ValueError("No documents loaded from DataFrame.")
    for doc, row in zip(docs, df.itertuples()):
        doc.metadata = {
            "credit_card_name": row.credit_card_name,
            "annual_fee": row.annual_fee
        }
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(faiss_index_dir)
else:
    vectorstore = FAISS.load_local(faiss_index_dir, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize LLM with Groq
llm = GroqLLM(
    client=groq_client,
    model="llama-3.3-70b-versatile",  # You can change to other Groq models
    temperature=0.2,
    max_tokens=2048
)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Utility functions
def get_card_details_by_name(card_name: str):
    try:
        card_name = card_name.strip().lower()
        matches = df[df["credit_card_name"].str.lower().str.strip() == card_name]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        partial_matches = df[df["credit_card_name"].str.lower().str.contains(card_name, na=False)]
        if not partial_matches.empty:
            best_partial = partial_matches.iloc[(partial_matches["credit_card_name"].str.len() - len(card_name)).abs().argmin()]
            print(f"Partial match found: {best_partial['credit_card_name']}")
            return best_partial.to_dict()
        
        # Fuzzy matching
        def all_words_in_name(name):
            words = card_name.split()
            return all(word in name.lower() for word in words)
        
        fuzzy_matches = df[df["credit_card_name"].apply(all_words_in_name)]
        if not fuzzy_matches.empty:
            best_fuzzy = fuzzy_matches.iloc[0]
            print(f"Fuzzy match found: {best_fuzzy['credit_card_name']}")
            return best_fuzzy.to_dict()
        
        return {"error": f"Card '{card_name}' not found."}
    except Exception as e:
        return {"error": f"Error retrieving card details: {e}"}

def compare_cards(query: str):
    try:
        # Extract card names from query
        cards = re.findall(r'(\w+(?:\s+\w+)*)', query)
        if len(cards) < 2:
            return {"error": "Please provide at least two card names to compare."}
        
        card1_name, card2_name = cards[0], cards[1]
        card1 = get_card_details_by_name(card1_name)
        card2 = get_card_details_by_name(card2_name)
        
        if "error" in card1 or "error" in card2:
            return {"error": f"Could not find one or both cards: {card1.get('error', '')} {card2.get('error', '')}"}
        
        return {"card1": card1, "card2": card2}
    except Exception as e:
        return {"error": f"Error comparing cards: {e}"}

def summarize_best_cards(input_data):
    try:
        # Parse input to extract cards
        if isinstance(input_data, str):
            # Try to extract card information from the string
            lines = input_data.split('\n')
            cards = []
            current_card = {}
            
            for line in lines:
                if line.strip().startswith('Card:'):
                    if current_card:
                        cards.append(current_card)
                    current_card = {'name': line.replace('Card:', '').strip()}
                elif ':' in line and current_card:
                    key, value = line.split(':', 1)
                    current_card[key.strip()] = value.strip()
            
            if current_card:
                cards.append(current_card)
        else:
            cards = input_data if isinstance(input_data, list) else []
        
        if not cards:
            return {"error": "No cards found to summarize."}
        
        # Process each card
        def clean_card(card):
            if isinstance(card, dict):
                return card
            return {"name": str(card)}
        
        def parse_fee(fee):
            if pd.isna(fee) or fee == 0:
                return "No Annual Fee"
            return f"₹{fee}"
        
        def parse_reward_rate(review):
            if pd.isna(review):
                return "Not specified"
            return str(review)[:100] + "..." if len(str(review)) > 100 else str(review)
        
        def parse_lounge_access(review):
            if pd.isna(review):
                return "Not specified"
            return str(review)[:100] + "..." if len(str(review)) > 100 else str(review)
        
        def parse_luxury_perks(review):
            if pd.isna(review):
                return "Not specified"
            return str(review)[:100] + "..." if len(str(review)) > 100 else str(review)
        
        def parse_welcome_bonus(review):
            if pd.isna(review):
                return "Not specified"
            return str(review)[:100] + "..." if len(str(review)) > 100 else str(review)
        
        # Create summary
        summary = f"Summary of Top {len(cards)} Credit Cards:\n\n"
        
        for i, card in enumerate(cards, 1):
            card = clean_card(card)
            name = card.get('name', 'Unknown Card')
            
            summary += f"{i}. {name}\n"
            summary += f"- Annual Fee: {parse_fee(card.get('annual_fee', 'N/A'))}\n"
            summary += f"- Reward Rate: {parse_reward_rate(card.get('cardmaven_review', 'N/A'))}\n"
            summary += f"- Lounge Access: {parse_lounge_access(card.get('cardexpert_review', 'N/A'))}\n"
            summary += f"- Luxury Perks: {parse_luxury_perks(card.get('best_tags', 'N/A'))}\n"
            summary += f"- Welcome Bonus: {parse_welcome_bonus(card.get('welcome_bonus', 'N/A'))}\n\n"
        
        # Add recommendation
        summary += "Recommendation:\n"
        summary += "Based on the analysis of these credit cards, I recommend considering the following factors:\n"
        summary += "1. Annual fees vs. benefits offered\n"
        summary += "2. Reward rates and cashback percentages\n"
        summary += "3. Lounge access and travel benefits\n"
        summary += "4. Welcome bonuses and sign-up offers\n"
        summary += "5. Your spending patterns and lifestyle needs\n\n"
        summary += "Please choose the card that best aligns with your financial goals and spending habits."
        
        return summary
    except Exception as e:
        return {"error": f"Error summarizing cards: {e}"}

def get_best_cards_by_tag(tag: str, max_fee=None, top_n: int = 5):
    try:
        tag = tag.strip().lower()
        filtered = df[df["best_tags"].str.contains(tag, case=False, na=False)]
        
        if not filtered.empty:
            if max_fee is not None:
                filtered = filtered[filtered["annual_fee"].fillna(99999) <= max_fee]
            if not filtered.empty:
                return filtered.head(top_n).to_dict(orient="records")
        
        # Try fuzzy matching
        all_tags = set()
        for tags in df["best_tags"].dropna():
            all_tags.update(tags.lower().split())
        
        close = difflib.get_close_matches(tag, list(all_tags), n=1, cutoff=0.5)
        if close:
            filtered = df[df["best_tags"].str.contains(close[0], case=False, na=False)]
            if not filtered.empty:
                if max_fee is not None:
                    filtered = filtered[filtered["annual_fee"].fillna(99999) <= max_fee]
                if not filtered.empty:
                    print(f"Using fuzzy-matched tag: {close[0]}")
                    return filtered.head(top_n).to_dict(orient="records")
        return [{"error": f"No cards found for tag '{tag}'."}]
    except Exception as e:
        return [{"error": f"Error retrieving cards for tag '{tag}': {e}"}]

def rag_lookup(query: str):
    try:
        return qa_chain.invoke({"query": query})["result"]
    except Exception as e:
        return f"Error processing query: {e}"

# Define tools for agent
TOOLS = [
    Tool(
    name="get_card_details_by_name",
    func=get_card_details_by_name,
    description="Get details about a specific credit card by name."
),
Tool(
    name="compare_cards_tool",
    func=compare_cards,
    description="Compare two credit cards. Input format: 'Card1 and Card2' or 'Card1 vs Card2'."
),
Tool(
    name="get_best_cards_by_tag",
    func=get_best_cards_by_tag,
    description="Get raw data of top cards in a category (e.g., 'best_cashback', 'best_travel')."
),
Tool(
    name="rag_lookup",
    func=rag_lookup,
    description="Answer general or vague credit card questions by retrieving relevant information."
),
Tool(
    name="summarize_best_cards",
    func=summarize_best_cards,
    description="Create a detailed summary and analysis of credit cards, including rankings and recommendations."
)

]

# Initialize agent
try:
    prefix = """You are a helpful assistant specialized in Indian credit cards.
    You have access to a set of tools that return structured or unstructured information.
    Your job is to understand the user's question, use tools if needed, and respond clearly.

    Only use the tools if absolutely necessary.

    After getting results from tools, ALWAYS respond with a Final Answer that is clear and helpful to the user, make it as elaborate as possible.
    """

    suffix = """Begin!

    Question: {input}
    {agent_scratchpad}"""

    prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=prefix + "\n\n" + suffix,
        )

    # ❗️Wrap in LLMChain — THIS IS MANDATORY
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # ZeroShotAgent needs an llm_chain
    agent_logic = ZeroShotAgent(llm_chain=llm_chain, tools=TOOLS)
    # Step 3: AgentExecutor with tools
    agent = AgentExecutor.from_agent_and_tools(
     agent=agent_logic,
     tools=TOOLS,
     verbose=True,
     handle_parsing_errors=True
    )

except Exception as e:
    raise Exception(f"Error initializing agent: {e}")

# Main handler
if __name__ == "__main__":
    print("Credit Card Assistant is running. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("Ask me anything about credit cards: ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            response = agent.invoke({"input": query})["output"]
            print("\nResponse:\n", response)
        except Exception as e:
            print(f"\nError processing query '{query}': {e}")

