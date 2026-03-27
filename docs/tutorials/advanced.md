# Advanced Examples

Creative and practical examples showcasing msgFlux capabilities.

---

## 🎙️ Podcast Episode Generator

Generate complete podcast episodes with AI hosts discussing any topic.

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")
tts = mf.Model.text_to_speech("openai/tts-1")


class TopicResearcher(nn.Agent):
    """Researches topics for podcast episodes."""
    model = model
    signature = """
    topic -> 
    key_points: list[str],
    interesting_facts: list[str],
    controversial_takes: list[str],
    expert_quotes: list[str]
    """


class ScriptWriter(nn.Agent):
    """Writes engaging podcast scripts."""
    model = model
    system_message = """
    You write natural, engaging podcast scripts for two hosts:
    - Alex: Enthusiastic, asks questions, uses analogies
    - Jordan: Analytical, provides depth, challenges assumptions
    """
    instructions = """
    Write a 5-minute podcast segment with:
    - Catchy intro hook
    - Natural banter between hosts
    - Key insights with examples
    - Surprising twist or revelation
    - Call-to-action outro
    
    Format as: [ALEX] or [JORDAN] followed by their lines.
    """
    message_fields = {"task_context": "research"}
    response_mode = "script"


class PodcastProducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.researcher = TopicResearcher()
        self.writer = ScriptWriter()
        self.alex_voice = nn.Speaker(
            model=mf.Model.text_to_speech("openai/tts-1", voice="alloy")
        )
        self.jordan_voice = nn.Speaker(
            model=mf.Model.text_to_speech("openai/tts-1", voice="onyx")
        )

    def forward(self, msg):
        # Research topic
        msg.research = self.researcher(msg.topic)
        
        # Write script
        self.writer(msg)
        
        # Parse and generate audio for each line
        lines = self._parse_script(msg.script)
        audio_segments = []
        
        for speaker, line in lines:
            voice = self.alex_voice if speaker == "ALEX" else self.jordan_voice
            audio = voice(line)
            audio_segments.append(audio)
        
        msg.audio_segments = audio_segments
        return msg

    def _parse_script(self, script):
        lines = []
        for line in script.split("\n"):
            if line.startswith("[ALEX]"):
                lines.append(("ALEX", line.replace("[ALEX]", "").strip()))
            elif line.startswith("[JORDAN]"):
                lines.append(("JORDAN", line.replace("[JORDAN]", "").strip()))
        return lines


# Usage
producer = PodcastProducer()

msg = Message()
msg.topic = "Why time might not exist"

producer(msg)

print("Script:")
print(msg.script)
print(f"\nGenerated {len(msg.audio_segments)} audio segments")
```

---

## 🎨 AI Art Director

Coordinate multiple specialists to create cohesive visual campaigns.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")
image_model = mf.Model.text_to_image("openai/dall-e-3")


class BrandAnalyst(nn.Agent):
    """Analyzes brand identity and visual language."""
    model = model
    signature = """
    brand_description, target_audience ->
    color_palette: list[str],
    visual_style: str,
    mood_keywords: list[str],
    typography_style: str,
    do_not_use: list[str]
    """


class ConceptArtist(nn.Agent):
    """Creates detailed image prompts."""
    model = model
    system_message = "You are an expert art director who creates detailed image prompts."
    instructions = """
    Create detailed image generation prompts that:
    - Match the brand's visual identity
    - Appeal to the target audience
    - Are technically specific (lighting, composition, style)
    - Include negative prompt suggestions
    """
    signature = """
    campaign_goal, brand_analysis ->
    hero_image_prompt: str,
    social_square_prompt: str,
    banner_wide_prompt: str,
    negative_prompt: str
    """


class ImageGenerator(nn.MediaMaker):
    model = image_model
    message_fields = {"task": "prompt"}


class ArtDirector(nn.Module):
    def __init__(self):
        super().__init__()
        self.analyst = BrandAnalyst()
        self.concepter = ConceptArtist()
        self.generator = ImageGenerator()

    def forward(self, msg):
        # Analyze brand
        msg.brand_analysis = self.analyst(
            brand_description=msg.brand_description,
            target_audience=msg.target_audience
        )
        
        # Create concepts
        msg.concepts = self.concepter(
            campaign_goal=msg.campaign_goal,
            brand_analysis=str(msg.brand_analysis)
        )
        
        # Generate hero image
        msg.prompt = msg.concepts.get("hero_image_prompt", "")
        self.generator.negative_prompt = msg.concepts.get("negative_prompt", "")
        self.generator(msg)
        
        return msg


# Usage
director = ArtDirector()

msg = Message()
msg.brand_description = "Sustainable coffee brand, artisanal, ethical sourcing"
msg.target_audience = "Environmentally conscious millennials, urban professionals"
msg.campaign_goal = "Launch new cold brew product for summer"

director(msg)

print("Brand Analysis:", msg.brand_analysis)
print("Concepts:", msg.concepts)
# msg.generated_image contains the hero image
```

---

## 🕵️ Competitive Intelligence Agent

Monitor competitors and generate strategic insights.

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


def scrape_website(url: str) -> str:
    """Scrape website content."""
    import requests
    from bs4 import BeautifulSoup
    
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        return soup.get_text(separator="\n")[:5000]
    except Exception as e:
        return f"Error: {e}"


class CompetitorAnalyzer(nn.Agent):
    """Analyzes competitor websites."""
    model = model
    tools = [scrape_website]
    signature = """
    competitor_url ->
    company_name: str,
    main_products: list[str],
    pricing_model: str,
    unique_selling_points: list[str],
    target_market: str,
    recent_updates: list[str]
    """


class ThreatAssessor(nn.Agent):
    """Assesses competitive threats."""
    model = model
    system_message = "You are a strategic analyst specializing in competitive intelligence."
    signature = """
    our_company, competitor_analyses ->
    direct_threats: list[dict],
    opportunities: list[dict],
    market_gaps: list[str],
    recommended_actions: list[str],
    urgency_level: Literal['low', 'medium', 'high', 'critical']
    """


class ReportGenerator(nn.Agent):
    """Generates executive reports."""
    model = model
    system_message = "You write concise executive briefings."
    instructions = """
    Create a one-page executive briefing with:
    - 3-sentence executive summary
    - Key metrics table
    - Top 3 action items with owners
    - Risk matrix (2x2)
    """


class IntelligenceSystem(nn.Module):
    def __init__(self, our_company: str):
        super().__init__()
        self.our_company = our_company
        self.analyzer = CompetitorAnalyzer()
        self.assessor = ThreatAssessor()
        self.reporter = ReportGenerator()

    def forward(self, msg):
        # Analyze each competitor in parallel
        analyses = F.map_gather(
            self.analyzer,
            args_list=[(url,) for url in msg.competitor_urls],
            kwargs_list=[{"competitor_url": url} for url in msg.competitor_urls]
        )
        
        msg.competitor_analyses = [a for a in analyses if a]
        
        # Assess threats
        msg.threat_assessment = self.assessor(
            our_company=self.our_company,
            competitor_analyses=str(msg.competitor_analyses)
        )
        
        # Generate report
        msg.executive_report = self.reporter(
            task_context={
                "analyses": msg.competitor_analyses,
                "assessment": msg.threat_assessment
            }
        )
        
        return msg


# Usage
intel = IntelligenceSystem(our_company="TechStartup Inc - AI Developer Tools")

msg = Message()
msg.competitor_urls = [
    "https://competitor1.com",
    "https://competitor2.com",
    "https://competitor3.com"
]

intel(msg)

print("Threat Assessment:", msg.threat_assessment)
print("\nExecutive Report:")
print(msg.executive_report)
```

---

## 📚 Personalized Learning Assistant

Adaptive tutor that adjusts to student's learning style.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class LearnerProfiler(nn.Agent):
    """Analyzes learning patterns."""
    model = model
    signature = """
    interaction_history ->
    learning_style: Literal['visual', 'auditory', 'kinesthetic', 'reading'],
    comprehension_speed: Literal['slow', 'moderate', 'fast'],
    preferred_examples: Literal['abstract', 'concrete', 'mixed'],
    knowledge_gaps: list[str],
    strengths: list[str]
    """


class ContentAdapter(nn.Agent):
    """Adapts content to learning style."""
    model = model
    system_message = "You adapt educational content to different learning styles."
    instructions = """
    Transform the content based on learner profile:
    
    - Visual: Use diagrams, flowcharts, color coding
    - Auditory: Conversational tone, mnemonics, rhythms
    - Kinesthetic: Hands-on examples, step-by-step, practice
    - Reading: Detailed text, references, structured notes
    
    Match complexity to comprehension speed.
    Use preferred example types.
    """


class QuizGenerator(nn.Agent):
    """Creates adaptive quizzes."""
    model = model
    signature = """
    topic, difficulty, learner_profile ->
    questions: list[dict],
    hints: list[str],
    explanations: list[str]
    """


class TutorAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.profiler = LearnerProfiler()
        self.adapter = ContentAdapter()
        self.quiz_gen = QuizGenerator()
        self.register_buffer("interaction_count", 0)

    def forward(self, msg):
        self.interaction_count += 1
        
        # Profile learner periodically
        if self.interaction_count % 5 == 1:
            msg.profile = self.profiler(
                interaction_history=str(msg.get("history", []))
            )
        
        # If asking about a topic
        if msg.get("learn_topic"):
            adapted = self.adapter(
                task=msg.learn_topic,
                task_context={"learner_profile": msg.get("profile", {})}
            )
            msg.lesson = adapted
            
        # If requesting quiz
        if msg.get("quiz_topic"):
            quiz = self.quiz_gen(
                topic=msg.quiz_topic,
                difficulty=msg.get("difficulty", "medium"),
                learner_profile=msg.get("profile", {})
            )
            msg.quiz = quiz
        
        return msg


# Usage
tutor = TutorAgent()

# Learning session
msg = Message()
msg.learn_topic = "How does photosynthesis work?"
msg.history = [
    "User asked about cells, needed visual diagram",
    "User understood mitochondria after analogy to power plant",
    "User struggled with abstract chemical formulas"
]

tutor(msg)
print("Learner Profile:", msg.profile)
print("\nAdapted Lesson:")
print(msg.lesson)

# Quiz
msg.quiz_topic = "photosynthesis"
msg.difficulty = "easy"
tutor(msg)
print("\nQuiz:", msg.quiz)
```

---

## 🎬 Movie Scene Analyzer

Analyze movie scenes for film students and critics.

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")  # Vision model


class VisualAnalyst(nn.Agent):
    """Analyzes visual composition."""
    model = model
    signature = """
    image ->
    composition: str,
    color_palette: list[str],
    lighting_type: str,
    camera_angle: str,
    depth_of_field: str,
    visual_metaphors: list[str]
    """


class NarrativeAnalyst(nn.Agent):
    """Analyzes narrative elements."""
    model = model
    signature = """
    image, context ->
    mood: str,
    tension_level: float,
    character_dynamics: str,
    foreshadowing: list[str],
    symbolic_elements: list[str]
    """


class FilmHistorian(nn.Agent):
    """Provides historical context."""
    model = model
    signature = """
    visual_analysis, narrative_analysis ->
    similar_films: list[str],
    director_influences: list[str],
    genre_conventions: list[str],
    innovation_notes: str
    """


class SceneAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisualAnalyst()
        self.narrative = NarrativeAnalyst()
        self.historian = FilmHistorian()

    def forward(self, msg):
        # Parallel visual and narrative analysis
        visual_result, narrative_result = F.bcast_gather(
            [
                lambda img: self.visual(
                    task_multimodal={"image": img}
                ),
                lambda img: self.narrative(
                    task_multimodal={"image": img},
                    context=msg.get("scene_context", "")
                )
            ],
            msg.scene_image
        )
        
        msg.visual_analysis = visual_result
        msg.narrative_analysis = narrative_result
        
        # Historical context
        msg.historical_context = self.historian(
            visual_analysis=str(visual_result),
            narrative_analysis=str(narrative_result)
        )
        
        return msg


# Usage
analyzer = SceneAnalyzer()

msg = Message()
msg.scene_image = "/path/to/movie_screenshot.jpg"
msg.scene_context = "Final confrontation scene, protagonist facing antagonist"

analyzer(msg)

print("Visual Analysis:", msg.visual_analysis)
print("\nNarrative Analysis:", msg.narrative_analysis)
print("\nHistorical Context:", msg.historical_context)
```

---

## 🏠 Smart Home Orchestrator

Voice-controlled home automation with natural language.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt = mf.Model.speech_to_text("openai/whisper-1")
tts = mf.Model.text_to_speech("openai/tts-1", voice="nova")


# Mock home devices
class SmartHomeAPI:
    @staticmethod
    def set_lights(room: str, brightness: int, color: str = "warm") -> str:
        return f"Lights in {room} set to {brightness}% {color}"
    
    @staticmethod
    def set_temperature(zone: str, temp: float) -> str:
        return f"Temperature in {zone} set to {temp}°C"
    
    @staticmethod
    def control_blinds(room: str, position: int) -> str:
        return f"Blinds in {room} set to {position}%"
    
    @staticmethod
    def play_music(playlist: str, room: str = "all") -> str:
        return f"Playing {playlist} in {room}"
    
    @staticmethod
    def set_scene(scene_name: str) -> str:
        scenes = {
            "movie": "Dimmed lights, blinds closed, surround on",
            "morning": "Gradual lights, blinds open, coffee starting",
            "party": "Color lights, music on, temperature cool"
        }
        return scenes.get(scene_name, f"Scene {scene_name} activated")


class HomeController(nn.Agent):
    """Controls smart home devices."""
    model = model
    system_message = """
    You are JARVIS, a helpful smart home assistant.
    Be concise, friendly, and proactive in suggestions.
    """
    tools = [
        SmartHomeAPI.set_lights,
        SmartHomeAPI.set_temperature,
        SmartHomeAPI.control_blinds,
        SmartHomeAPI.play_music,
        SmartHomeAPI.set_scene
    ]
    config = {"verbose": True}


class VoiceInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = nn.Transcriber(model=stt)
        self.controller = HomeController()
        self.speaker = nn.Speaker(model=tts)

    def forward(self, msg):
        # Transcribe if audio input
        if msg.get("voice_input"):
            msg.content = self.transcriber(msg.voice_input)
        
        # Process command
        msg.response = self.controller(msg.content)
        
        # Generate voice response
        msg.voice_output = self.speaker(msg.response)
        
        return msg


# Usage
home = VoiceInterface()

# Text command
msg = Message()
msg.content = "Set the living room for movie night and make it cozy"
home(msg)
print(msg.response)

# Voice command
msg = Message()
msg.voice_input = "/path/to/voice_command.wav"
home(msg)
print(msg.content)  # Transcription
print(msg.response)  # Action result
# msg.voice_output = audio response
```

---

## 🔮 Startup Idea Validator

Validate startup ideas with multi-perspective analysis.

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class VCPartner(nn.Agent):
    """Evaluates from VC perspective."""
    model = model
    system_message = """
    You are a seasoned VC partner at a top-tier firm.
    You've seen 10,000 pitches and invested in 50 unicorns.
    Be direct, skeptical, but fair.
    """
    signature = """
    idea ->
    market_size_assessment: str,
    defensibility: Literal['weak', 'moderate', 'strong'],
    team_requirements: list[str],
    investment_thesis: str,
    deal_breakers: list[str],
    would_take_meeting: bool
    """


class SerialEntrepreneur(nn.Agent):
    """Evaluates from founder perspective."""
    model = model
    system_message = """
    You've built and exited 3 startups.
    You know execution matters more than ideas.
    Focus on practical challenges.
    """
    signature = """
    idea ->
    execution_roadblocks: list[str],
    first_90_days_plan: list[str],
    bootstrap_potential: bool,
    pivot_options: list[str],
    founder_market_fit: str,
    advice: str
    """


class DevilsAdvocate(nn.Agent):
    """Finds all possible failure modes."""
    model = model
    system_message = """
    Your job is to find every way this could fail.
    Be ruthless but constructive.
    """
    signature = """
    idea ->
    failure_scenarios: list[dict],
    competitor_threats: list[str],
    regulatory_risks: list[str],
    timing_concerns: list[str],
    why_hasnt_this_been_done: str
    """


class CustomerProxy(nn.Agent):
    """Represents target customer perspective."""
    model = model
    system_message = """
    You represent the target customer.
    Think: Would I actually pay for this? Would I switch from current solution?
    """
    signature = """
    idea ->
    pain_point_validation: str,
    willingness_to_pay: Literal['none', 'low', 'medium', 'high'],
    switching_barriers: list[str],
    must_have_features: list[str],
    deal_breakers: list[str]
    """


class StartupValidator(nn.Module):
    def __init__(self):
        super().__init__()
        self.vc = VCPartner()
        self.founder = SerialEntrepreneur()
        self.devil = DevilsAdvocate()
        self.customer = CustomerProxy()

    def forward(self, msg):
        # Get all perspectives in parallel
        vc, founder, devil, customer = F.bcast_gather(
            [self.vc, self.founder, self.devil, self.customer],
            msg.idea
        )
        
        msg.vc_analysis = vc
        msg.founder_analysis = founder
        msg.risk_analysis = devil
        msg.customer_analysis = customer
        
        # Calculate overall score
        msg.validation_score = self._calculate_score(vc, founder, devil, customer)
        
        return msg

    def _calculate_score(self, vc, founder, devil, customer):
        score = 0
        if vc.get("would_take_meeting"): score += 25
        if vc.get("defensibility") == "strong": score += 15
        if founder.get("bootstrap_potential"): score += 15
        if customer.get("willingness_to_pay") == "high": score += 25
        if len(devil.get("failure_scenarios", [])) < 3: score += 20
        return score


# Usage
validator = StartupValidator()

msg = Message()
msg.idea = """
AI-powered legal document analyzer for small businesses.
Reads contracts and flags risky clauses in plain English.
$49/month subscription. Target: Startups and freelancers
who can't afford lawyers for every contract review.
"""

validator(msg)

print(f"Validation Score: {msg.validation_score}/100")
print("\n📊 VC Perspective:", msg.vc_analysis)
print("\n🚀 Founder Perspective:", msg.founder_analysis)
print("\n⚠️ Risk Analysis:", msg.risk_analysis)
print("\n👤 Customer Perspective:", msg.customer_analysis)
```
