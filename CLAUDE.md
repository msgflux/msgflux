# CLAUDE.md - msgFlux Project Context

**Last Updated**: 2026-01-18
**Version**: 0.1.0a20
**Status**: Active Development (Alpha/Beta)

---

## Overview

**msgFlux** is a comprehensive Python library for building sophisticated AI systems powered by pretrained models. It provides a PyTorch-like API for composing AI agents, tools, and workflows with support for multiple model providers, advanced telemetry, and production-ready features.

### Core Philosophy

- **Modular Architecture**: Composable components that work together seamlessly
- **Provider Agnostic**: Support for multiple AI providers (OpenAI, Google, Anthropic, etc.)
- **Production Ready**: Built-in telemetry, caching, retry logic, and error handling
- **Developer Friendly**: Intuitive API inspired by PyTorch and modern ML frameworks

---

## Architecture

### Core Components

msgFlux is organized into several major subsystems:

#### 1. Models Layer (`src/msgflux/models/`)

Unified interface for AI model providers with support for:
- **Chat Completion** - Conversational AI
- **Embeddings** - Vector representations
- **Text-to-Speech** - Audio generation
- **Speech-to-Text** - Transcription
- **Image Generation** - Visual content creation
- **Moderation** - Content filtering

**Supported Providers** (12+):
- OpenAI
- Groq
- Cerebras 
- Ollama
- Together AI
- OpenRouter
- SambaNova
- vLLM
- JinaAI

**Model Profiles**:
- Integration with models.dev for automatic model discovery
- Local caching of model metadata
- TTL-based cache invalidation
- Fallback to local profiles when offline

#### 2. Neural Network Modules (`src/msgflux/nn/`)

PyTorch-inspired API for composing AI systems:

**Core Modules**:
- `Module` - Base class for all neural modules (similar to torch.nn.Module)
- `Agent` - Autonomous AI agents with tool use and reasoning
- `Tool` - Function calling and external tool integration
- `LM` - Language model wrapper
- `Embedder` - Text embedding models
- `Searcher` - Document retrieval systems
- `Speaker` - Text-to-speech synthesis
- `Transcriber` - Speech-to-text transcription
- `MediaMaker` - Image/video generation
- `Sequential` - Chain multiple modules
- `ModuleDict` - Dictionary of modules
- `ModuleList` - List of modules

**Special Features**:
- Automatic telemetry for all modules
- Built-in caching at module level
- Composability through nesting
- Parameter management like PyTorch
- Support for async/await patterns

#### 3. Data Layer (`src/msgflux/data/`)

Comprehensive data handling infrastructure:

**Parsers** (`data/parsers/providers/`):
- CSV - Tabular data parsing
- HTML - Web content extraction
- Markdown - Document parsing
- Email - Email message parsing
- PDF - Document extraction (pypdf)
- DOCX - Word documents (python-docx)
- PPTX - PowerPoint (python-pptx)
- XLSX - Excel spreadsheets (openpyxl)

**Databases** (`data/dbs/providers/`):
- CacheTools - In-memory caching
- DiskCache - Persistent disk-based cache
- FAISS - Vector similarity search

**Retrievers** (`data/retrievers/providers/`):
- BM25 - Classic ranking algorithm
- RankBM25 - Optimized BM25 implementation
- Wikipedia - Wikipedia article retrieval

#### 4. DSL (Domain-Specific Language) (`src/msgflux/dsl/`)

Declarative interface for AI workflows:

**Signature System**:
- `InputField` - Define input schemas
- `OutputField` - Define output schemas
- `Signature` - Type-safe function signatures
- `Audio`, `Image`, `Video`, `File` - Media types

**Inline Functions**:
- `inline` - Synchronous inline AI functions
- `ainline` - Async inline AI functions
- Automatic prompt generation from docstrings
- Type-safe input/output validation

**Typed Parsers** (`dsl/typed_parsers/`):
- XML parsing with schema validation
- Future: Integration with txml (Rust) for performance
- Future: TSN format parsing with tsn (Rust)

#### 5. Protocols (`src/msgflux/protocols/`)

**Model Context Protocol (MCP)** (`protocols/mcp/`):
- Full MCP specification implementation
- Multiple transport types:
  - `StdioTransport` - Standard I/O communication
  - `HTTPTransport` - HTTP-based communication
  - `BaseTransport` - Custom transport base class

**Authentication Methods** (`protocols/mcp/auth/`):
- `BasicAuth` - HTTP Basic Authentication
- `BearerTokenAuth` - OAuth 2.0 Bearer tokens
- `APIKeyAuth` - API key authentication
- `OAuth2Auth` - Full OAuth 2.0 flow
- `CustomHeaderAuth` - Custom header-based auth
- `BaseAuth` - Custom authentication base class

**MCP Features**:
- Tool discovery and registration
- Resource management
- Prompt templates
- Server lifecycle management
- Error handling and retries
- Async/await support

#### 6. Telemetry (`src/msgflux/telemetry/`)

Production-grade observability powered by **msgtrace-sdk v1.1.0+**:

**Core Features**:
- OpenTelemetry-based tracing
- GenAI semantic conventions
- Automatic span creation for all operations
- Zero overhead when disabled
- Thread-safe with async support

**Span Types**:
- `Spans.flow()` - High-level workflow tracking
- `Spans.module()` - Module execution tracking
- `Spans.operation()` - Low-level operation tracking
- `Spans.instrument()` - Decorator-based instrumentation

**Specialized Tracking**:
- Tool execution (local/remote, MCP)
- Agent conversations and reasoning
- Model API calls with token counts
- Retrieval operations
- Custom user-defined spans

**Configuration**:
- Environment-based configuration
- OTLP export to Jaeger, Zipkin, etc.
- Console exporter for debugging
- Batch processing for efficiency

#### 7. Generation (`src/msgflux/generation/`)

Advanced text generation patterns:

**Reasoning Strategies** (`generation/reasoning/`):
- `CoT` - Chain of Thought reasoning
- `ReAct` - Reasoning + Acting pattern
- `SelfConsistency` - Multiple reasoning paths with voting

**Templates** (`generation/templates.py`):
- Jinja2-based prompt templates
- Built-in template library
- Custom template support

**Control Flow** (`generation/control_flow.py`):
- Conditional generation
- Iterative refinement
- Multi-step workflows

#### 8. Utilities (`src/msgflux/utils/`)

Comprehensive utility library:

- `chat.py` - ChatML and conversation formatting
- `msgspec.py` - Fast JSON serialization with msgspec
- `encode.py` - Token counting and encoding
- `console.py` - Rich console output (cprint)
- `inspect.py` - Function introspection
- `hooks.py` - Lifecycle hooks
- `imports.py` - Dynamic import utilities
- `logging.py` - Structured logging
- `mermaid.py` - Diagram generation
- `pooling.py` - Text pooling strategies
- `tenacity.py` - Retry decorators
- `torch.py` - PyTorch utilities
- `validation.py` - Input validation
- `xml.py` - XML helpers
- `common.py` - Common utilities
- `convert.py` - Type conversion

---

## Project Structure

```
msgflux/
├── .github/                    # GitHub configuration
│   ├── workflows/              # CI/CD workflows
│   │   ├── ci.yml             # Continuous integration
│   │   ├── docs.yml           # Documentation deployment
│   │   ├── publish.yml        # PyPI publishing
│   │   ├── validate-release.yml  # Release security validation
│   │   ├── merge-bot.yml      # Automated PR merging
│   │   ├── labeler.yml        # Automatic PR labeling
│   │   ├── release-drafter.yml   # Release notes generation
│   │   └── pre-commit-autoupdate.yml  # Pre-commit updates
│   ├── CODEOWNERS             # Code ownership definitions
│   ├── labeler.yml            # Labeler configuration
│   ├── pull_request_template.md  # PR template
│   └── release-drafter.yml    # Release drafter config
│
├── src/msgflux/               # Main source code
│   ├── __init__.py            # Public API exports
│   ├── version.py             # Version information
│   │
│   ├── _private/              # Internal implementations
│   │   ├── client.py          # HTTP client wrapper
│   │   ├── core.py            # Core utilities
│   │   ├── executor.py        # Async execution
│   │   └── response.py        # Response handling
│   │
│   ├── data/                  # Data handling
│   │   ├── dbs/               # Database backends
│   │   │   ├── base.py        # Base DB interface
│   │   │   ├── db.py          # DB registry
│   │   │   └── providers/     # DB implementations
│   │   ├── parsers/           # Document parsers
│   │   │   ├── base.py        # Base parser
│   │   │   ├── parser.py      # Parser registry
│   │   │   └── providers/     # Parser implementations
│   │   └── retrievers/        # Retrieval systems
│   │       ├── base.py        # Base retriever
│   │       ├── retriever.py   # Retriever registry
│   │       └── providers/     # Retriever implementations
│   │
│   ├── dsl/                   # Domain-specific language
│   │   ├── inline.py          # Inline function decorators
│   │   ├── signature.py       # Type signatures
│   │   └── typed_parsers/     # Typed parsing (XML, TSN)
│   │
│   ├── generation/            # Text generation
│   │   ├── control_flow.py    # Control flow patterns
│   │   ├── templates.py       # Prompt templates
│   │   └── reasoning/         # Reasoning strategies
│   │       ├── cot.py         # Chain of Thought
│   │       ├── react.py       # ReAct pattern
│   │       └── self_consistency.py  # Self-consistency
│   │
│   ├── models/                # Model providers
│   │   ├── base.py            # Base model interface
│   │   ├── model.py           # Model registry
│   │   ├── gateway.py         # ModelGateway implementation
│   │   ├── cache.py           # Response caching
│   │   ├── httpx.py           # HTTP client
│   │   ├── profiles/          # Model profiles system
│   │   │   ├── base.py        # Profile base classes
│   │   │   ├── loader.py      # Profile loading
│   │   │   └── registry.py    # Profile registry
│   │   └── providers/         # Provider implementations
│   │       ├── openai.py      # OpenAI (GPT-4, GPT-3.5)
│   │       ├── cerebras.py    # Cerebras inference
│   │       ├── groq.py        # Groq fast inference
│   │       ├── ollama.py      # Ollama local models
│   │       ├── together.py    # Together AI
│   │       ├── replicate.py   # Replicate
│   │       ├── vllm.py        # vLLM self-hosted
│   │       ├── sambanova.py   # SambaNova
│   │       ├── openrouter.py  # OpenRouter gateway
│   │       ├── jinaai.py      # JinaAI embeddings
│   │       └── imagerouter.py # Image generation routing
│   │
│   ├── nn/                    # Neural network modules
│   │   ├── __init__.py        # Module exports
│   │   ├── functional.py      # Functional API
│   │   ├── parameter.py       # Parameter management
│   │   └── modules/           # Module implementations
│   │       ├── module.py      # Base Module class
│   │       ├── agent.py       # Agent module
│   │       ├── tool.py        # Tool system
│   │       ├── lm.py          # Language model
│   │       ├── embedder.py    # Embeddings
│   │       ├── searcher.py    # Retrieval
│   │       ├── speaker.py     # Text-to-speech
│   │       ├── transcriber.py # Speech-to-text
│   │       ├── mediamaker.py  # Media generation
│   │       └── container.py   # Container modules
│   │
│   ├── protocols/             # Protocol implementations
│   │   └── mcp/               # Model Context Protocol
│   │       ├── __init__.py    # MCP exports
│   │       ├── client.py      # MCP client
│   │       ├── exceptions.py  # MCP exceptions
│   │       ├── integration.py # Integration helpers
│   │       ├── loglevels.py   # Logging configuration
│   │       ├── transports.py  # Transport implementations
│   │       ├── types.py       # Type definitions
│   │       └── auth/          # Authentication
│   │           ├── base.py    # Base auth
│   │           ├── basic.py   # Basic auth
│   │           ├── bearer.py  # Bearer token
│   │           ├── apikey.py  # API key
│   │           ├── oauth2.py  # OAuth 2.0
│   │           └── custom.py  # Custom auth
│   │
│   ├── telemetry/             # Observability
│   │   ├── __init__.py        # Telemetry exports
│   │   └── span.py            # Span decorators
│   │
│   ├── tools/                 # Tool system
│   │   └── config.py          # Tool configuration
│   │
│   ├── utils/                 # Utilities
│   │   ├── chat.py            # Chat formatting
│   │   ├── msgspec.py         # JSON serialization
│   │   ├── encode.py          # Token encoding
│   │   ├── console.py         # Console output
│   │   ├── inspect.py         # Introspection
│   │   ├── hooks.py           # Lifecycle hooks
│   │   ├── imports.py         # Dynamic imports
│   │   ├── logging.py         # Logging
│   │   ├── mermaid.py         # Diagrams
│   │   ├── pooling.py         # Pooling
│   │   ├── tenacity.py        # Retries
│   │   ├── torch.py           # PyTorch utils
│   │   ├── validation.py      # Validation
│   │   ├── xml.py             # XML helpers
│   │   ├── common.py          # Common utils
│   │   └── convert.py         # Type conversion
│   │
│   ├── cache.py               # Response caching
│   ├── dotdict.py             # Dictionary utilities
│   ├── envs.py                # Environment management
│   ├── examples.py            # Example management
│   ├── exceptions.py          # Exception definitions
│   ├── logger.py              # Logger setup
│   ├── message.py             # Message types
│   └── py.typed               # PEP 561 marker
│
├── tests/                     # Test suite
│   └── models/
│       └── test_profiles.py   # Model profiles tests
│
├── docs/                      # MkDocs documentation
│   ├── index.md               # Documentation home
│   ├── quickstart.md          # Quick start guide
│   ├── api-reference/         # API documentation
│   ├── blog/                  # Blog posts
│   ├── getting-started/       # Tutorials
│   ├── guides/                # How-to guides
│   └── stylesheets/           # Custom CSS
│
├── scripts/                   # Automation scripts
│   ├── release.sh             # Release automation
│   ├── setup-branch-protection.sh  # Branch protection
│   └── setup-labels.sh        # GitHub labels
│
├── .pre-commit-config.yaml    # Pre-commit hooks
├── .python-version            # Python version
├── .readthedocs.yaml          # ReadTheDocs config
├── CHANGELOG.md               # Changelog
├── CLAUDE.md                  # This file
├── CONTRIBUTING.md            # Contributing guide
├── README.md                  # Project README
├── mkdocs.yml                 # MkDocs configuration
├── pyproject.toml             # Project metadata
└── uv.lock                    # Dependency lock file
```

**File Statistics**:
- Total Python files: 143 in `src/`
- Test files: Growing coverage
- Documentation: Comprehensive MkDocs site
- Workflows: 8 GitHub Actions workflows

---

## Dependencies

### Core Dependencies

Required for all installations:

```toml
jinja2>=3.1.6              # Templating engine
msgspec-ext>=0.5.0         # Environment management & validation
msgtrace-sdk>=1.1.0        # OpenTelemetry tracing
tenacity>=8.2.3            # Retry logic with exponential backoff
typing-extensions>=4.14.1  # Python typing backports
uvloop>=0.21.0             # Fast event loop (Unix only)
```

### Optional Dependencies

Install via `pip install msgflux[provider]`:

**Provider Groups**:
- `openai` - OpenAI models (GPT-4, DALL-E, Whisper)
  - openai>=1.97.1
  - opentelemetry-instrumentation-openai>=0.43.1

**Feature Groups**:
- `httpx` - HTTP client support
  - httpx>=0.28.1

- `plot` - Diagram generation
  - code2mermaid>=0.3.0
  - mermaid-py>=0.8.0

### Development Groups

Install via `uv sync --group <group>`:

**dev** - Development tools:
```toml
pytest>=8.4.1              # Testing framework
pytest-asyncio>=1.1.0      # Async test support
pytest-cov>=6.2.1          # Coverage reporting
pytest-mock>=3.14.0        # Mocking utilities
ruff>=0.12.5               # Linting and formatting
twine>=6.0.1               # Package publishing
packaging>=24.0            # Version parsing
```

**doc** - Documentation:
```toml
markdown-include>=0.8.1    # Markdown file inclusion
mkdocs>=1.6.1              # Documentation generator
mkdocs-material>=9.6.21    # Material theme
mkdocstrings>=0.30.1       # API doc generation
mkdocstrings-python>=1.18.2  # Python docstring parsing
pymdown-extensions>=10.16.1  # Markdown extensions
```

**msgtrace** - Observability service development:
```toml
fastapi>=0.109.0           # API framework
uvicorn[standard]>=0.27.0  # ASGI server
pydantic>=2.0.0            # Data validation
websockets>=12.0           # WebSocket support
aiohttp>=3.9.0             # Async HTTP
python-multipart>=0.0.6    # Multipart parsing
httpx>=0.28.1              # HTTP client
```

### Dependency Management

- **Tool**: UV (ultra-fast Python package manager)
- **Lock File**: uv.lock (committed to repository)
- **Reproducibility**: Exact versions locked for CI/CD
- **Updates**: Dependabot automated updates weekly

---

## Development Workflow

### Setup

```bash
# Clone repository
git clone https://github.com/msgflux/msgflux.git
cd msgflux

# Install dependencies
uv sync --group dev --group doc

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest -v

# Run documentation server
uv run --group doc mkdocs serve
```

### Code Quality Tools

**Ruff** - All-in-one linter and formatter:
```bash
# Format code
uv run ruff format

# Check formatting
uv run ruff format --check

# Lint code
uv run ruff check

# Auto-fix issues
uv run ruff check --fix
```

**Configuration**:
- Target: Python 3.10+
- Line length: 88 characters
- Select: Comprehensive rule set (A, B, C, D, E, F, etc.)
- Ignore: Specific relaxations for developer experience

**Pre-commit Hooks**:
- gitleaks - Secret detection
- ruff-format - Code formatting
- ruff-lint - Linting
- uv-lock - Dependency lock validation

### Branch Strategy

- **main** - Production-ready code, protected branch
- **Feature branches** - `feat/feature-name`, `fix/bug-name`, etc.
- **Conventional naming**:
  - feat/ - New features
  - fix/ - Bug fixes
  - docs/ - Documentation
  - refactor/ - Code refactoring
  - test/ - Tests
  - chore/ - Maintenance
  - perf/ - Performance

### Commit Convention

Following [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

### Release Process

Automated via `scripts/release.sh`:

```bash
./scripts/release.sh 0.2.0
```

**Process**:
1. Version validation (semantic versioning)
2. Update version.py and CHANGELOG.md
3. Security validation (only 2 files modified)
4. Create release branch
5. Create PR with labels
6. CI validation
7. Manual merge to trigger publish
8. Automated PyPI upload
9. GitHub release creation

---

## CI/CD Infrastructure

### GitHub Actions Workflows

**ci.yml** - Continuous Integration:
- Runs on: PR, push to main
- Python: 3.10, 3.11, 3.12, 3.13
- Steps:
  1. Ruff format check
  2. Ruff lint
  3. Pytest with coverage
  4. Build distribution

**docs.yml** - Documentation Deployment:
- Runs on: Push to main
- Builds: MkDocs site
- Deploys: GitHub Pages (https://msgflux.github.io/msgflux/)

**publish.yml** - PyPI Publishing:
- Trigger: Version change on main
- Validates: Version bump
- Builds: Source and wheel distributions
- Publishes: PyPI via trusted publishing
- Creates: Git tag and GitHub release

**validate-release.yml** - Security Validation:
- Runs on: All commits to main
- Validates: Only version.py and CHANGELOG.md modified
- Purpose: Prevent supply chain attacks

**merge-bot.yml** - Automated Merging:
- Commands: `/merge`, `/update`
- Checks: CI status, approvals
- Action: Squash and merge

**labeler.yml** - Auto-labeling:
- Analyzes: Changed files
- Applies: Relevant labels (models, core, tests, docs, etc.)

**release-drafter.yml** - Release Notes:
- Generates: Changelog from PRs
- Categorizes: Features, fixes, etc.

**pre-commit-autoupdate.yml** - Hook Updates:
- Schedule: Weekly
- Updates: Pre-commit hook versions
- Creates: PR with updates

### Branch Protection

**Rules for main**:
- Require PR before merge
- Require status checks to pass
- Enforce for administrators
- Require linear history (squash merge)
- Require conversation resolution
- No force pushes
- No deletions

**Required Checks**:
- CI / Ruff Lint & Format
- CI / Test Python 3.10, 3.11, 3.12, 3.13
- CI / Build distribution
- Validate Release / Validate Only Release Files Changed

### Dependabot

- **Schedule**: Weekly updates
- **Target**: GitHub Actions, pip dependencies
- **Auto-merge**: Patch and minor updates
- **Groups**: Related dependencies together

---

## API Reference

### Public API (`msgflux.*`)

```python
from msgflux import (
    # Core
    Model,           # Model interface
    ModelGateway,    # Multi-provider gateway
    Message,         # Message type
    Example,         # Few-shot example

    # Data
    DB,              # Database interface
    Retriever,       # Retriever interface

    # DSL
    Signature,       # Type signature
    InputField,      # Input field definition
    OutputField,     # Output field definition
    inline,          # Sync inline function
    ainline,         # Async inline function

    # Media Types
    Audio,           # Audio type
    Image,           # Image type
    Video,           # Video type
    File,            # File type

    # Utils
    ChatML,          # Chat formatting
    ChatBlock,       # Chat block
    dotdict,         # Enhanced dict
    cprint,          # Colored print
    get_fn_name,     # Function name

    # Serialization
    load,            # Load JSON
    save,            # Save JSON
    msgspec_dumps,   # Fast JSON dumps

    # Telemetry
    Spans,           # Tracing spans

    # Configuration
    set_envs,        # Environment setup
    tool_config,     # Tool configuration
    response_cache,  # Response caching
)
```

### Neural Network API (`msgflux.nn.*`)

```python
from msgflux.nn import (
    # Base
    Module,          # Base module class
    Parameter,       # Learnable parameter

    # Agents
    Agent,           # Autonomous agent

    # Language
    LM,              # Language model

    # Tools
    Tool,            # Function tool
    LocalTool,       # Local function tool
    MCPTool,         # MCP protocol tool
    ToolLibrary,     # Tool collection

    # Retrieval
    Searcher,        # Retrieval module
    Embedder,        # Embedding module

    # Media
    Speaker,         # Text-to-speech
    Transcriber,     # Speech-to-text
    MediaMaker,      # Image/video generation

    # Containers
    Sequential,      # Sequential container
    ModuleDict,      # Dict container
    ModuleList,      # List container

    # Functional API
    functional,      # Functional operations
)
```

### Model Context Protocol (`msgflux.protocols.mcp.*`)

```python
from msgflux.protocols.mcp import (
    # Client
    MCPClient,       # MCP client

    # Transports
    StdioTransport,  # Stdio transport
    HTTPTransport,   # HTTP transport
    BaseTransport,   # Base transport

    # Authentication
    BasicAuth,       # Basic auth
    BearerTokenAuth, # Bearer token
    APIKeyAuth,      # API key
    OAuth2Auth,      # OAuth 2.0
    CustomHeaderAuth,# Custom headers
    BaseAuth,        # Base auth class

    # Exceptions
    MCPError,        # Base MCP error
    MCPConnectionError,  # Connection error
    MCPTimeoutError, # Timeout error
    MCPToolError,    # Tool execution error

    # Utilities
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,

    # Types
    LogLevel,        # Log level enum
    MCPContent,      # Content type
)
```

---

## Testing

### Test Suite

**Framework**: pytest with async support

**Structure**:
```
tests/
├── models/
│   └── test_profiles.py
├── nn/
│   └── test_modules.py
└── ...
```

**Running Tests**:
```bash
# All tests
uv run pytest -v

# With coverage
uv run pytest -v --cov=src/msgflux --cov-report=html

# Specific test
uv run pytest tests/models/test_profiles.py -v

# Fast (no coverage)
uv run pytest
```

---

## Documentation

### MkDocs Site

**Live**: https://msgflux.github.io/msgflux/

**Structure**:
- Homepage - Project overview
- Quickstart - Get started quickly
- API Reference - Complete API docs
- Guides - How-to guides
- Blog - Updates and tutorials

**Technology**:
- MkDocs - Static site generator
- Material Theme - Modern, responsive design
- mkdocstrings - API documentation from docstrings
- Python-Markdown - Extended Markdown

**Building Locally**:
```bash
# Install doc dependencies
uv sync --group doc

# Serve locally
uv run --group doc mkdocs serve

# Build static site
uv run --group doc mkdocs build
```

**Deployment**:
- Automatic via GitHub Actions
- Trigger: Push to main
- Target: gh-pages branch
- URL: https://msgflux.github.io/msgflux/

---

### Development Guidelines

**DO**:
- ✅ Follow conventional commit messages
- ✅ Add type hints to all public APIs
- ✅ Write docstrings for public functions
- ✅ Add tests for new features
- ✅ Update CHANGELOG.md for notable changes
- ✅ Run ruff before committing
- ✅ Use uv for dependency management
- ✅ Keep PRs focused and small

**DON'T**:
- ❌ Push directly to main
- ❌ Modify msgtrace-sdk core (use as dependency)
- ❌ Add breaking changes without discussion
- ❌ Skip CI checks
- ❌ Commit without running tests
- ❌ Add dependencies without justification
- ❌ Use `git add -A` (stage files explicitly)

### Key Files to Know

- `src/msgflux/__init__.py` - Public API exports
- `src/msgflux/version.py` - Version (auto-updated by release script)
- `pyproject.toml` - Project metadata and dependencies
- `CHANGELOG.md` - Changelog (update with PRs)
- `CONTRIBUTING.md` - Development guide
- `mkdocs.yml` - Documentation configuration
