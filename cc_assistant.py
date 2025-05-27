
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
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent
# Hardcoded Together AI API key (replace with your key)
TOGETHER_API_KEY = "f53ce47139359ade99476808912d868305efc505d0071eb40f0fcbc0f8f9742b"
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set.")

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

# Initialize LLM with Together AI
llm = ChatTogether(
    api_key=TOGETHER_API_KEY,
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
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
        words = card_name.split()
        def all_words_in_name(name):
            return all(word in name for word in words)
        token_matches = df[df["credit_card_name"].str.lower().apply(all_words_in_name)]
        if not token_matches.empty:
            best_token = token_matches.iloc[(token_matches["credit_card_name"].str.len() - len(card_name)).abs().argmin()]
            print(f"Token match found: {best_token['credit_card_name']}")
            return best_token.to_dict()
        names = df["credit_card_name"].str.lower().tolist()
        close = difflib.get_close_matches(card_name, names, n=1, cutoff=0.4)
        if close:
            match_row = df[df["credit_card_name"].str.lower() == close[0]]
            if not match_row.empty:
                print(f"Fuzzy match found: {close[0]}")
                return match_row.iloc[0].to_dict()
        print(f"No card found for '{card_name}'.")
        return f"No card found for '{card_name}'."
    except Exception as e:
        return f"Error retrieving card details for '{card_name}': {e}"

def compare_cards(query: str):
    try:
        pattern = re.compile(r"\s*(?:and|vs|versus)\s*", re.IGNORECASE)
        cards = pattern.split(query)
        if len(cards) != 2:
            return f"Invalid comparison format. Please use 'Card1 and Card2' or 'Card1 vs Card2'."
        card1, card2 = [c.strip() for c in cards]
        return {
            "card_1": get_card_details_by_name(card1),
            "card_2": get_card_details_by_name(card2)
        }
    except Exception as e:
        return f"Error comparing cards: {e}"

def summarize_best_cards(input_data):
    try:
        import ast
        tag = "unknown"
        cards = []
        if isinstance(input_data, str):
            try:
                data = ast.literal_eval(input_data)
            except Exception:
                return f"Error: Could not parse input data."
        else:
            data = input_data
        if isinstance(data, dict):
            tag = data.get("tag", "unknown")
            cards = data.get("cards", [])
        elif isinstance(data, list):
            cards = data
        else:
            return "Error: Input data format not recognized."
        if not cards or (isinstance(cards, dict) and "error" in cards):
            return f"No cards found for tag '{tag}' to summarize."
        weights = {
            "annual_fee": 0.3,
            "reward_rate": 0.3,
            "lounge_access": 0.2,
            "luxury_perks": 0.15,
            "welcome_bonus": 0.05
        }
        allowed_fields = {"credit_card_name", "annual_fee", "cardmaven_review", "cardexpert_review", "best_tags"}
        def clean_card(card):
            return {k: v for k, v in card.items() if k in allowed_fields}
        cards = [clean_card(card) for card in cards]
        def parse_fee(fee):
            if isinstance(fee, str):
                fee = fee.replace("₹", "").replace(",", "").replace(" + GST", "").strip()
                try:
                    return float(fee)
                except ValueError:
                    return 99999
            return float(fee) if isinstance(fee, (int, float)) else 99999
        def parse_reward_rate(review):
            if isinstance(review, str):
                match = re.search(r"reward rate:?\s*(\d+\.?\d*%\s*-\s*\d+\.?\d*%)", review, re.IGNORECASE)
                if match:
                    low, high = map(lambda x: float(x.strip("%")), match.group(1).split("-"))
                    return (low + high) / 2
            return 1.0
        def parse_lounge_access(review):
            if isinstance(review, str):
                if "unlimited" in review.lower() and "lounge" in review.lower():
                    return 3
                if "lounge access" in review.lower():
                    return 2
            return 1
        def parse_luxury_perks(review):
            if not isinstance(review, str):
                return 0
            perks = 0
            if "concierge" in review.lower():
                perks += 1
            if "golf" in review.lower():
                perks += 1
            if any(x in review.lower() for x in ["accor", "marriott", "taj epicure"]):
                perks += 1
            return perks
        def parse_welcome_bonus(review):
            if isinstance(review, str):
                match = re.search(r"welcome benefit:?\s*(\d{1,3}(?:,\d{3})*)\s*(?:points|inr)", review, re.IGNORECASE)
                if match:
                    return float(match.group(1).replace(",", ""))
            return 0
        scored_cards = []
        for card in cards:
            fee = parse_fee(card.get("annual_fee", "99999"))
            review = card.get("cardmaven_review", "") or card.get("cardexpert_review", "")
            reward_rate = parse_reward_rate(review)
            lounge_score = parse_lounge_access(review)
            perks_score = parse_luxury_perks(review)
            bonus_score = parse_welcome_bonus(review)
            fee_score = max(0, 1 - (fee / 50000))
            reward_score = min(reward_rate / 20, 1)
            lounge_score = lounge_score / 3
            perks_score = perks_score / 3
            bonus_score = min(bonus_score / 20000, 1)
            total_score = (
                weights["annual_fee"] * fee_score +
                weights["reward_rate"] * reward_score +
                weights["lounge_access"] * lounge_score +
                weights["luxury_perks"] * perks_score +
                weights["welcome_bonus"] * bonus_score
            )
            scored_cards.append({
                "name": card.get("credit_card_name", "Unknown"),
                "score": total_score,
                "annual_fee": fee,
                "reward_rate": reward_rate,
                "lounge_access": lounge_score * 3,
                "luxury_perks": perks_score * 3,
                "welcome_bonus": bonus_score * 20000
            })
        scored_cards.sort(key=lambda x: x["score"], reverse=True)
        summary = f"Summary of Top {len(scored_cards)} Premium Credit Cards:\n\n"
        for i, card in enumerate(scored_cards, 1):
            summary += (
                f"{i}. {card['name']}\n"
                f"   - Annual Fee: ₹{card['annual_fee']:,.0f}\n"
                f"   - Reward Rate: ~{card['reward_rate']:.1f}%\n"
                f"   - Lounge Access: {'Unlimited' if card['lounge_access'] >= 3 else 'Limited' if card['lounge_access'] >= 2 else 'None'}\n"
                f"   - Luxury Perks: {int(card['luxury_perks'])}/3 (e.g., concierge, golf, hotel memberships)\n"
                f"   - Welcome Bonus: ~₹{int(card['welcome_bonus']):,.0f} in points/vouchers\n"
                f"   - Score: {card['score']:.2f}/1.00\n\n"
            )
        top_card = scored_cards[0]
        summary += (
            f"Recommendation:\n"
            f"The {top_card['name']} is the top premium card with a score of {top_card['score']:.2f}, "
            f"offering a strong balance of a {top_card['reward_rate']:.1f}% reward rate, "
            f"{'unlimited' if top_card['lounge_access'] >= 3 else 'limited'} lounge access, "
            f"and {int(top_card['luxury_perks'])} luxury perks. "
            f"Its ₹{top_card['annual_fee']:,.0f} annual fee is justified for high spenders seeking premium benefits."
        )
        if len(scored_cards) > 1 and scored_cards[1]["score"] > 0.8 * top_card["score"]:
            summary += (
                f" The {scored_cards[1]['name']} is a close second with a score of {scored_cards[1]['score']:.2f}, "
                f"especially if you prefer a lower fee or specific perks."
            )
        return summary
    except Exception as e:
        return f"Error summarizing cards: {e}"

def get_best_cards_by_tag(tag: str, max_fee=None, top_n: int = 5):
    try:
        filtered = df[df["best_tags"].str.contains(tag, case=False, na=False)]
        if not filtered.empty:
            if max_fee is not None:
                filtered = filtered[filtered["annual_fee"].fillna(99999) <= max_fee]
            if not filtered.empty:
                return filtered.head(top_n).to_dict(orient="records")
        all_tags = set()
        for tags in df["best_tags"].dropna().unique():
            for t in str(tags).split(","):
                all_tags.add(t.strip().lower())
        tag_words = set(tag.replace("best_", "").split("_"))
        best_match = None
        best_score = 0
        for known_tag in all_tags:
            known_words = set(known_tag.replace("best_", "").split("_"))
            overlap = len(tag_words & known_words)
            if overlap > best_score or (overlap == best_score and tag in known_tag):
                best_score = overlap
                best_match = known_tag
        if best_match and best_score > 0:
            filtered = df[df["best_tags"].str.contains(best_match, case=False, na=False)]
            if not filtered.empty:
                if max_fee is not None:
                    filtered = filtered[filtered["annual_fee"].fillna(99999) <= max_fee]
                if not filtered.empty:
                    print(f"Using best-overlap tag: {best_match}")
                    return filtered.head(top_n).to_dict(orient="records")
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

