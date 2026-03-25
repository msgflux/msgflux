# Tutorials

Step-by-step tutorials inspired by DSPy, showing how to build production-ready AI applications with msgFlux.

---

## RAG (Retrieval-Augmented Generation)

Build a question-answering system that retrieves relevant context before generating answers.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message

# Setup
model = mf.Model.chat_completion("openai/gpt-4.1-mini")
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")
vectordb = mf.DataBase.vector("faiss")


# Index documents
documents = [
    "Python was created by Guido van Rossum and released in 1991.",
    "JavaScript was created by Brendan Eich in 1995 at Netscape.",
    "Rust was first released in 2010, created by Graydon Hoare at Mozilla.",
    "Go was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson.",
    "TypeScript is a superset of JavaScript developed by Microsoft in 2012.",
]

# Embed and store
embeddings = []
for doc in documents:
    emb = embedder(doc).consume().data
    embeddings.append(emb)

vectordb.add([{"text": doc, "embedding": emb} for doc, emb in zip(documents, embeddings)])


class RAGRetriever(nn.Searcher):
    """Retrieves relevant documents."""
    retriever = vectordb
    model = embedder
    message_fields = {"task_inputs": "question"}
    response_mode = "context"


class RAGGenerator(nn.Agent):
    """Generates answer from context."""
    model = model
    system_message = "Answer questions using only the provided context."
    instructions = "If the answer is not in the context, say you don't know."
    message_fields = {
        "task_inputs": "question",
        "context_inputs": "context"
    }
    response_mode = "answer"


class RAG(nn.Module):
    def __init__(self):
        super().__init__()
        self.retriever = RAGRetriever()
        self.generator = RAGGenerator()

    def forward(self, msg):
        self.retriever(msg)
        self.generator(msg)
        return msg


# Usage
rag = RAG()

msg = Message()
msg.question = "Who created Python?"

rag(msg)

print(f"Question: {msg.question}")
print(f"Context: {msg.context}")
print(f"Answer: {msg.answer}")
```

---

## Entity Extraction

Extract structured information from unstructured text.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message
from typing import Optional

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class PersonExtractor(nn.Agent):
    """Extracts person information from text."""
    model = model
    signature = """
    text ->
    name: str,
    age: Optional[int],
    occupation: Optional[str],
    location: Optional[str],
    email: Optional[str],
    phone: Optional[str]
    """


class CompanyExtractor(nn.Agent):
    """Extracts company information from text."""
    model = model  
    signature = """
    text ->
    company_name: str,
    industry: Optional[str],
    founded_year: Optional[int],
    headquarters: Optional[str],
    employee_count: Optional[str],
    revenue: Optional[str]
    """


class EventExtractor(nn.Agent):
    """Extracts event information from text."""
    model = model
    signature = """
    text ->
    event_name: str,
    date: Optional[str],
    location: Optional[str],
    organizer: Optional[str],
    description: str
    """


# Usage
person_extractor = PersonExtractor()
company_extractor = CompanyExtractor()

text1 = """
John Smith is a 35-year-old software engineer based in San Francisco.
He can be reached at john.smith@techcorp.com or (415) 555-0123.
"""

result = person_extractor(text1)
print("Person:", result)
# {'name': 'John Smith', 'age': 35, 'occupation': 'software engineer', 
#  'location': 'San Francisco', 'email': 'john.smith@techcorp.com', 
#  'phone': '(415) 555-0123'}

text2 = """
TechCorp Inc., founded in 2015, is a leading AI company headquartered in 
Palo Alto, California. With over 500 employees, they generated $50M in 
revenue last year.
"""

result = company_extractor(text2)
print("Company:", result)
```

---

## Classification

Build a text classifier with structured outputs.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message
from typing import Literal

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class SentimentClassifier(nn.Agent):
    """Classifies text sentiment."""
    model = model
    signature = """
    text -> 
    sentiment: Literal['positive', 'negative', 'neutral'],
    confidence: float,
    key_phrases: list[str]
    """


class IntentClassifier(nn.Agent):
    """Classifies user intent for customer support."""
    model = model
    signature = """
    message ->
    intent: Literal['billing', 'technical_support', 'product_inquiry', 
                    'complaint', 'general_question', 'cancellation'],
    urgency: Literal['low', 'medium', 'high'],
    requires_human: bool,
    summary: str
    """


class TopicClassifier(nn.Agent):
    """Classifies document topics."""
    model = model
    signature = """
    document ->
    primary_topic: str,
    secondary_topics: list[str],
    keywords: list[str],
    reading_level: Literal['elementary', 'high_school', 'college', 'expert']
    """


# Usage
sentiment = SentimentClassifier()
intent = IntentClassifier()
topic = TopicClassifier()

# Sentiment
result = sentiment("I absolutely love this product! It exceeded all my expectations.")
print("Sentiment:", result)

# Intent
result = intent("My subscription was charged twice this month. I need a refund immediately!")
print("Intent:", result)

# Topic
doc = """
Quantum computing leverages quantum mechanical phenomena like superposition 
and entanglement to perform computations. Unlike classical bits, qubits can 
exist in multiple states simultaneously, enabling exponential speedups for 
certain algorithms like Shor's factoring algorithm.
"""
result = topic(doc)
print("Topic:", result)
```

---

## Multi-Hop RAG

Answer complex questions requiring multiple retrieval steps.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message
from msgflux.generation.reasoning import ReAct

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    # Mock implementation
    results = {
        "python creator": "Guido van Rossum created Python in the late 1980s.",
        "guido van rossum": "Guido van Rossum was born in Netherlands in 1956.",
        "netherlands": "The Netherlands is a country in Western Europe.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return "No results found."


def search_knowledge_base(query: str) -> str:
    """Search internal knowledge base."""
    return f"Knowledge base results for: {query}"


class MultiHopAgent(nn.Agent):
    """Agent that performs multi-hop reasoning."""
    model = model
    
    system_message = """
    You are a research assistant that answers complex questions.
    Break down complex questions into simpler sub-questions.
    Use tools to find information step by step.
    """
    
    tools = [search_wikipedia, search_knowledge_base]
    generation_schema = ReAct
    
    templates = {"response": "{{final_answer}}"}
    config = {"verbose": True}


# Usage
agent = MultiHopAgent()

# Question requiring multiple hops
question = """
What is the capital of the country where the creator of Python was born?
"""

response = agent(question)
print("Answer:", response)
# Should trace: Python creator → Guido van Rossum → Netherlands → Amsterdam
```

---

## Customer Service Agent

Build an intelligent customer service chatbot.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# Mock tools
def check_order_status(order_id: str) -> str:
    """Check the status of an order."""
    orders = {
        "ORD-123": "Shipped - Expected delivery: Tomorrow",
        "ORD-456": "Processing - Will ship in 2 days",
        "ORD-789": "Delivered on Jan 15, 2024"
    }
    return orders.get(order_id, "Order not found")


def initiate_refund(order_id: str, reason: str) -> str:
    """Initiate a refund for an order."""
    return f"Refund initiated for {order_id}. Reason: {reason}. Processing time: 3-5 days."


def check_product_availability(product_name: str) -> str:
    """Check if a product is in stock."""
    return f"{product_name} is in stock. 15 units available."


def escalate_to_human(issue_summary: str) -> str:
    """Escalate complex issues to human support."""
    return f"Ticket created. A human agent will contact you within 24 hours. Issue: {issue_summary}"


class IntentRouter(nn.Agent):
    """Classifies customer intent."""
    model = model
    signature = """
    message, conversation_history ->
    intent: Literal['order_status', 'refund_request', 'product_inquiry', 
                    'complaint', 'general', 'escalate'],
    extracted_order_id: Optional[str],
    extracted_product: Optional[str]
    """


class CustomerServiceAgent(nn.Agent):
    """Main customer service agent."""
    model = model
    
    system_message = """
    You are a friendly and helpful customer service agent for TechStore.
    
    Guidelines:
    - Be empathetic and professional
    - Solve problems efficiently
    - Proactively offer additional help
    - Escalate to human when unsure
    """
    
    tools = [check_order_status, initiate_refund, 
             check_product_availability, escalate_to_human]
    
    config = {"verbose": True}


class CustomerServiceBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = IntentRouter()
        self.agent = CustomerServiceAgent()

    def forward(self, msg):
        # Route intent
        intent = self.router(
            message=msg.content,
            conversation_history=str(msg.get("history", []))
        )
        msg.intent = intent
        
        # Handle with context
        context = f"Customer intent: {intent.get('intent')}"
        if intent.get("extracted_order_id"):
            context += f"\nOrder ID: {intent['extracted_order_id']}"
        
        msg.response = self.agent(
            msg.content,
            context_inputs=context
        )
        
        return msg


# Usage
bot = CustomerServiceBot()

# Conversation
msg = Message()
msg.content = "Hi, I ordered something last week, order number ORD-123. Where is it?"
msg.history = []

bot(msg)

print(f"Intent: {msg.intent}")
print(f"Response: {msg.response}")
```

---

## Conversation History

Manage multi-turn conversations with context.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class ConversationalAgent(nn.Agent):
    """Agent that maintains conversation context."""
    model = model
    system_message = "You are a helpful assistant. Remember previous context."
    config = {"return_model_state": True}


# Conversation loop using ChatML
chat = mf.ChatML()
agent = ConversationalAgent()

# Turn 1
user_msg = "Hi! My name is Alice and I'm a software engineer."
chat.add_user_message(user_msg)

response = agent(task_messages=chat.get_messages())
chat.add_assist_message(response.model_response)
print(f"User: {user_msg}")
print(f"Agent: {response.model_response}\n")

# Turn 2
user_msg = "What frameworks should I learn for web development?"
chat.add_user_message(user_msg)

response = agent(task_messages=chat.get_messages())
chat.add_assist_message(response.model_response)
print(f"User: {user_msg}")
print(f"Agent: {response.model_response}\n")

# Turn 3 - Agent should remember name and profession
user_msg = "Which one would be best for someone like me?"
chat.add_user_message(user_msg)

response = agent(task_messages=chat.get_messages())
print(f"User: {user_msg}")
print(f"Agent: {response.model_response}")

# View full conversation
print("\n--- Full Conversation ---")
for msg in chat.get_messages():
    print(f"{msg['role'].upper()}: {msg['content']}")
```

---

## Email Information Extraction

Extract structured data from emails.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message
from typing import Optional, List

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class MeetingExtractor(nn.Agent):
    """Extracts meeting information from emails."""
    model = model
    signature = """
    email_body ->
    is_meeting_request: bool,
    proposed_times: list[str],
    duration_minutes: Optional[int],
    location: Optional[str],
    virtual_link: Optional[str],
    attendees: list[str],
    agenda_items: list[str]
    """


class ActionItemExtractor(nn.Agent):
    """Extracts action items from emails."""
    model = model
    signature = """
    email_body ->
    action_items: list[dict],
    deadlines: list[dict],
    priority_level: Literal['low', 'medium', 'high', 'urgent'],
    requires_response: bool,
    response_deadline: Optional[str]
    """


class EmailSummarizer(nn.Agent):
    """Summarizes email threads."""
    model = model
    signature = """
    email_thread ->
    main_topic: str,
    key_decisions: list[str],
    open_questions: list[str],
    tldr: str
    """


class EmailProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.meeting = MeetingExtractor()
        self.actions = ActionItemExtractor()
        self.summarizer = EmailSummarizer()

    def forward(self, msg):
        email = msg.email_body
        
        msg.meeting_info = self.meeting(email)
        msg.action_items = self.actions(email)
        msg.summary = self.summarizer(email)
        
        return msg


# Usage
processor = EmailProcessor()

email = """
From: Sarah Johnson <sarah@company.com>
Subject: Q1 Planning Meeting - Action Required

Hi Team,

I'd like to schedule our Q1 planning session for next week. 
Here are the proposed times:
- Tuesday, Jan 21 at 2:00 PM EST
- Wednesday, Jan 22 at 10:00 AM EST

The meeting will be 90 minutes via Zoom: https://zoom.us/j/123456

Agenda:
1. Review Q4 results
2. Set Q1 OKRs
3. Resource allocation
4. Budget discussion

Please confirm your availability by Friday EOD.

@Mike - please prepare the Q4 sales report
@Lisa - bring the customer feedback summary
@Tom - update the roadmap slides

Looking forward to it!
Sarah
"""

msg = Message()
msg.email_body = email

processor(msg)

print("Meeting Info:", msg.meeting_info)
print("\nAction Items:", msg.action_items)
print("\nSummary:", msg.summary)
```

---

## Financial Analysis with Tools

Build a financial analysis agent with market data tools.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message
from datetime import datetime

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# Mock financial tools
def get_stock_price(symbol: str) -> dict:
    """Get current stock price and daily change."""
    prices = {
        "AAPL": {"price": 185.50, "change": 2.3, "volume": "45M"},
        "GOOGL": {"price": 142.80, "change": -0.5, "volume": "22M"},
        "MSFT": {"price": 378.20, "change": 1.8, "volume": "18M"},
    }
    return prices.get(symbol.upper(), {"error": f"Symbol {symbol} not found"})


def get_company_financials(symbol: str) -> dict:
    """Get company financial metrics."""
    return {
        "market_cap": "2.85T",
        "pe_ratio": 28.5,
        "eps": 6.51,
        "revenue_growth": "8.5%",
        "profit_margin": "24.3%"
    }


def get_analyst_ratings(symbol: str) -> dict:
    """Get analyst ratings and price targets."""
    return {
        "buy": 25,
        "hold": 8,
        "sell": 2,
        "average_target": 195.00,
        "high_target": 220.00,
        "low_target": 165.00
    }


def get_market_news(topic: str) -> list:
    """Get recent market news."""
    return [
        {"title": "Tech stocks surge on AI optimism", "source": "Reuters"},
        {"title": "Fed signals rate stability", "source": "Bloomberg"},
    ]


class FinancialAnalyst(nn.Agent):
    """AI-powered financial analyst."""
    model = model
    
    system_message = """
    You are a professional financial analyst.
    Provide data-driven insights and balanced analysis.
    Always cite your sources and note any limitations.
    Do not provide investment advice.
    """
    
    tools = [get_stock_price, get_company_financials, 
             get_analyst_ratings, get_market_news]
    
    config = {"verbose": True}


class StockReporter(nn.Agent):
    """Generates stock reports."""
    model = model
    signature = """
    stock_symbol, analysis_data ->
    summary: str,
    strengths: list[str],
    risks: list[str],
    key_metrics: dict,
    outlook: Literal['bullish', 'bearish', 'neutral']
    """


# Usage
analyst = FinancialAnalyst()
reporter = StockReporter()

# Analysis query
response = analyst("Give me a comprehensive analysis of Apple (AAPL) stock")
print("Analysis:", response)

# Structured report
report = reporter(
    stock_symbol="AAPL",
    analysis_data=response
)
print("\nReport:", report)
```

---

## Audio Processing

Work with audio inputs for transcription and analysis.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt = mf.Model.speech_to_text("openai/whisper-1")
tts = mf.Model.text_to_speech("openai/tts-1")


class AudioTranscriber(nn.Transcriber):
    """Transcribes audio to text."""
    model = stt
    response_mode = "content"


class TranscriptAnalyzer(nn.Agent):
    """Analyzes transcripts."""
    model = model
    signature = """
    transcript ->
    summary: str,
    main_topics: list[str],
    action_items: list[str],
    sentiment: Literal['positive', 'negative', 'neutral'],
    key_quotes: list[str]
    """


class MeetingNoteTaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = AudioTranscriber()
        self.analyzer = TranscriptAnalyzer()

    def forward(self, msg):
        # Transcribe
        if msg.get("audio_path"):
            msg.content = self.transcriber(msg.audio_path)
        
        # Analyze
        msg.analysis = self.analyzer(msg.content)
        
        return msg


# Usage  
note_taker = MeetingNoteTaker()

msg = Message()
msg.audio_path = "/path/to/meeting_recording.mp3"

note_taker(msg)

print("Transcript:", msg.content[:500], "...")
print("\nAnalysis:", msg.analysis)
```

---

## Streaming Responses

Handle streaming responses for real-time output.

```python
import msgflux as mf
import msgflux.nn as nn
import asyncio

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class StreamingAgent(nn.Agent):
    """Agent with streaming responses."""
    model = model
    system_message = "You are a storyteller. Write engaging stories."
    config = {"stream": True}


async def main():
    agent = StreamingAgent()
    
    # Get streaming response
    response = agent("Write a short story about a robot learning to paint")
    
    # Stream chunks
    print("Streaming story:\n")
    full_text = ""
    async for chunk in response.consume():
        print(chunk, end="", flush=True)
        full_text += chunk
    
    print(f"\n\nTotal length: {len(full_text)} characters")


# Run
asyncio.run(main())
```

---

## Async Concurrent Execution

Process multiple requests concurrently.

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
import asyncio

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class QuickAnalyzer(nn.Agent):
    """Fast single-purpose analyzer."""
    model = model


async def analyze_batch(texts: list[str]) -> list:
    """Analyze multiple texts concurrently."""
    agent = QuickAnalyzer()
    
    # Create tasks
    tasks = [agent.acall(text) for text in texts]
    
    # Run concurrently
    results = await asyncio.gather(*tasks)
    return results


# Usage
texts = [
    "The product quality exceeded my expectations!",
    "Terrible customer service, never buying again.",
    "It's okay, nothing special.",
    "Best purchase I've made this year!",
    "Would not recommend to others."
]

async def main():
    results = await analyze_batch(texts)
    
    for text, result in zip(texts, results):
        print(f"Text: {text[:50]}...")
        print(f"Analysis: {result}\n")

asyncio.run(main())
```
