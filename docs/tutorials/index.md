# Examples

A collection of creative examples demonstrating msgFlux capabilities.

---

## 🎙️ Podcast Generator

An automated pipeline that converts a topic into a two-host podcast episode, complete with script writing, voice synthesis, and audio mixing.

**Concepts**: `nn.Sequential`, `nn.Speaker`, `msg_bcast_gather`, `AutoParams`

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message

# Models
chat_model = mf.Model.chat_completion("openai/gpt-4")
tts_model = mf.Model.text_to_speech("openai/tts-1")

class ScriptWriter(nn.Agent):
    """Writes a dynamic dialogue between two hosts."""
    model = chat_model
    system_message = "You are a professional podcast scriptwriter."
    instructions = """
    Write a 2-minute dialogue between two hosts: 
    - Alex (Enthusiastic, main host)
    - Jamie (Skeptical but curious, co-host)
    
    Topic: {{topic}}
    
    Output the script as a list of turns.
    """
    signature = "topic -> script: list[dict[str, str]]"
    message_fields = {"task": "topic"}
    response_mode = "script"

class AudioProducer(nn.Module):
    """Synthesizes audio for each turn in parallel."""
    def __init__(self):
        super().__init__()
        # Define speakers
        self.alex = nn.Speaker(
            model=tts_model, 
            config={"voice": "alloy", "speed": 1.0}
        )
        self.jamie = nn.Speaker(
            model=tts_model, 
            config={"voice": "onyx", "speed": 1.05}
        )

    def forward(self, msg):
        script = msg.script
        
        def process_turn(turn):
            speaker_name = turn['speaker'].lower()
            text = turn['text']
            
            # Select voice
            speaker = self.alex if 'alex' in speaker_name else self.jamie
            
            # Generate audio in parallel
            return speaker(text)

        # Generate all lines concurrently
        msg.audio_segments = F.map_gather(process_turn, [(turn,) for turn in script])
        return msg

class AudioMixer(nn.Module):
    """Combines audio segments (mock implementation)."""
    def forward(self, msg):
        # In a real app, use pydub to concatenate audio bytes
        msg.final_podcast = b"".join(msg.audio_segments)
        return msg

# Podcast Pipeline
podcast_gen = nn.Sequential(
    ScriptWriter(),
    AudioProducer(),
    AudioMixer()
)

# Run
msg = Message(topic="The Future of Space Travel")
podcast_gen(msg)

# Save
with open("podcast.mp3", "wb") as f:
    f.write(msg.final_podcast)
```

---

## 🎨 AI Art Director

A chain of agents that refine prompts, generate images, and critique them iteratively.

**Concepts**: `ModuleDict`, `inline` DSL, `Conditional Logic`

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F

class ArtDirector(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleDict({
            # Ideation
            "conceptualizer": nn.Agent(
                model=mf.Model.chat_completion("openai/gpt-4"),
                instructions="Turn this abstract concept into 3 visual prompt ideas.",
                response_mode="concepts"
            ),
            # Production
            "artist": nn.MediaMaker(
                model=mf.Model.text_to_image("openai/dall-e-3"),
                message_fields={"task": "selected_concept"},
                response_mode="artwork"
            ),
            # Quality Control
            "critic": nn.Agent(
                model=mf.Model.chat_completion("openai/gpt-4-vision-preview"),
                instructions="Rate this image 1-10 on composition and relevance.",
                signature="image -> score: int, critique: str",
                message_fields={"task_multimodal": {"image": "artwork"}},
                response_mode="review"
            )
        })
        
        # Define the workflow DSL
        self.register_buffer("workflow", """
        conceptualizer 
        -> select_best 
        -> artist 
        -> critic 
        -> {score < 7 ? retry : finalize}
        """)

    def select_best(self, msg):
        # Simple logic to pick the first concept
        msg.selected_concept = msg.concepts[0]
        return msg

    def finalize(self, msg):
        msg.status = "approved"
        return msg
        
    def retry(self, msg):
        msg.status = "rejected"
        # Update concept based on critique for next loop
        msg.selected_concept = f"{msg.selected_concept}. Improve: {msg.review['critique']}"
        # Recurse (simplified)
        return self.models["artist"](msg)

    def forward(self, msg):
        # Add local methods to the execution scope
        scope = dict(self.models)
        scope.update({
            "select_best": self.select_best,
            "finalize": self.finalize,
            "retry": self.retry
        })
        
        return F.inline(self.workflow, scope, msg)
```

---

## 🏠 Intelligent Home Hub

A routing system that directs user commands to specialized IoT agents.

**Concepts**: `ModuleDict`, `Routing`, `Intent Classification`

```python
class IntentRouter(nn.Agent):
    """Classifies user intent to route to the correct subsystem."""
    model = mf.Model.chat_completion("openai/gpt-4-turbo")
    signature = "command -> system: Literal['lighting', 'hvac', 'security', 'media']"
    
class HomeHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = IntentRouter()
        
        # Subsystems
        self.systems = nn.ModuleDict({
            "lighting": nn.Agent(instructions="Convert command to Zigbee JSON for lights."),
            "hvac": nn.Agent(instructions="Convert command to thermostat adjustments."),
            "security": nn.Agent(instructions="Handle locks and cameras."),
            "media": nn.Agent(instructions="Control TV and speakers.")
        })

    def forward(self, msg):
        # 1. Determine intent
        intent = self.router(msg.command)
        msg.system = intent['system']
        
        # 2. Route dynamically
        if msg.system in self.systems:
            # Execute specific agent
            self.systems[msg.system](msg)
        else:
            msg.error = "Unknown system"
            
        return msg

hub = HomeHub()
result = hub(mf.Message(command="Dim the lights and set temperature to 72"))
# Routes to 'lighting' -> (handling logic needed for multiple intents)
```

---

## 🚀 Startup Idea Validator

A comprehensive analysis pipeline running parallel market research.

**Concepts**: `bcast_gather`, `Parallel Execution`, `Aggregation`

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F

class Validator(nn.Module):
    def __init__(self):
        super().__init__()
        model = mf.Model.chat_completion("openai/gpt-4")
        
        # Parallel Analysts
        self.analysts = [
            nn.Agent(model, instructions="Analyze market size and growth CAGR.", name="market"),
            nn.Agent(model, instructions="Identify 3 key competitors and their weaknesses.", name="competitors"),
            nn.Agent(model, instructions="List technical feasibility risks.", name="tech"),
            nn.Agent(model, instructions="Suggest monetization strategies.", name="finance")
        ]
        
        self.synthesizer = nn.Agent(
            model, 
            instructions="Synthesize all reports into a Go/No-Go recommendation.",
            message_fields={"task_context": "reports"}
        )

    def forward(self, msg):
        idea = msg.idea
        
        # Run all analysts in parallel
        # Each returns a distinct report string
        reports = F.bcast_gather(self.analysts, idea)
        
        # Store results
        msg.reports = {
            "market": reports[0],
            "competitors": reports[1],
            "tech": reports[2],
            "finance": reports[3]
        }
        
        # Synthesize final decision
        self.synthesizer(msg)
        return msg

validator = Validator()
res = validator(mf.Message(idea="Uber for dog walking"))
print(res.content) # Final recommendation
```

---

## 📚 Personalized Learning Assistant

Adapts content difficulty based on user feedback loop.

**Concepts**: `State Management`, `Conditionals`, `Adaptive Logic`

```python
class Tutor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mf.Model.chat_completion("openai/gpt-4")
        
        # State tracking (would exist in DB in prod)
        self.register_buffer("level", 1) 
        
    def generate_lesson(self, topic):
        return nn.Agent(
            self.model,
            instructions=f"Explain {{topic}} at difficulty level {self.level}/5."
        )(topic)
        
    def check_understanding(self, response):
        return nn.Agent(
            self.model,
            signature="response -> score: int"
        )(response)

    def forward(self, msg):
        # 1. Teach
        msg.lesson = self.generate_lesson(msg.topic)
        
        # 2. Wait for user input (simulated here)
        msg.user_response = input(f"Lesson: {msg.lesson}\nWhat did you understand? ")
        
        # 3. Assess
        assessment = self.check_understanding(msg.user_response)
        
        # 4. Adapt State
        if assessment['score'] > 8:
            self.level = min(5, self.level + 1)
            msg.feedback = "Great! Moving to next level."
        elif assessment['score'] < 4:
            self.level = max(1, self.level - 1)
            msg.feedback = "Let's review the basics."
        else:
            msg.feedback = "Good, let's practice more."
            
        return msg
```
