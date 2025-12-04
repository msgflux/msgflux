# Claude.md - msgFlux Project Context

## Overview

**msgFlux** is a Python library for building AI systems powered by pretrained models. The project has grown significantly and is undergoing architectural restructuring to separate components into isolated projects for better management.

## Project Information

- **Repository**: vilsonrodrigues/msgflux (fork from msgflux/msgflux)
- **Current Branch**: `init`
- **Upstream**: msgflux/msgflux
- **Python Version**: 3.9+
- **Status**: Active development, Beta stage

## Architecture Evolution

### Separated Components

The project is being split into specialized libraries:

#### 1. msgspec-ext (Stable ✓)
- **Purpose**: Environment variable management and type validation
- **Status**: Already stable and integrated as dependency
- **Key Features**: Best-in-class Python library for dotenv reading and type validation
- **Integration**: Already listed in pyproject.toml dependencies
- **Note**: Not to be confused with msgtrace - this is msgspec-ext

#### 2. msgtrace (In Development)
- **Purpose**: Isolated observability service
- **Components**:
  - Backend for trace collection
  - Frontend dashboard
  - SDK (msgtrace-sdk) with similar interface to msgflux's current implementation
- **Status**: Currently being developed separately
- **Migration Plan**: Will replace built-in OpenTelemetry observability in msgflux

#### 3. txml (Rust Implementation)
- **Purpose**: XML parsing
- **Status**: Replaces Python-based XML parser implementation
- **Language**: Rust
- **Note**: Instead of referencing msgflux's Python XML implementation, we now reference txml

#### 4. tsn (Rust Implementation)
- **Purpose**: Custom format parsing (tsn format)
- **Language**: Rust
- **Status**: New Rust-based implementation

### Current Built-in Features (To Be Migrated)

The msgflux codebase still includes:
- OpenTelemetry-based observability (src/msgflux/telemetry/)
  - Will be replaced by msgtrace integration
- Python XML parsers (src/msgflux/dsl/typed_parsers/)
  - Will be replaced by txml references

## Project Structure

```
msgflux/
├── src/msgflux/
│   ├── data/              # Data handling, databases, retrievers
│   ├── dsl/               # Domain-specific language features
│   │   └── typed_parsers/ # To be replaced by txml/tsn
│   ├── generation/        # Text generation utilities
│   ├── models/            # Model providers and interfaces
│   │   └── providers/     # OpenAI, Google, Anthropic, Mistral, etc.
│   ├── nn/                # Neural network modules
│   ├── protocols/         # Protocol implementations
│   │   └── mcp/           # Model Context Protocol
│   ├── telemetry/         # OpenTelemetry integration (to be replaced by msgtrace)
│   ├── tools/             # Tool definitions
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docs/                  # Documentation (currently in .gitignore)
├── examples/              # Example code (currently in .gitignore)
├── next_version/          # Next version development (currently in .gitignore)
└── .github/               # CI/CD workflows
```

## Current State & Git Status

### Uncommitted Changes
Many files remain uncommitted, particularly in:
- `docs/` - Documentation files (ignored for now)
- `examples/` - Example code (ignored for now)
- `next_version/` - Future version development (ignored for now)

These directories are temporarily in .gitignore while we stabilize the core.

### Recent Changes
- CI/CD improvements (will be migrated from init branch to upstream)
- New model providers: Anthropic, Google, Mistral
- MCP (Model Context Protocol) integration
- Telemetry improvements
- Pre-commit hooks setup

## Dependencies

### Core Dependencies
- `msgspec-ext>=0.1.0` - Environment management
- `opentelemetry-api>=1.35.0` - Observability (temporary, will migrate to msgtrace)
- `opentelemetry-sdk>=1.35.0`
- `opentelemetry-exporter-otlp>=1.35.0`
- `jinja2>=3.1.6` - Templating
- `tenacity>=8.2.3` - Retry logic
- `uvloop>=0.21.0` - Event loop (Unix only)

### Optional Dependencies
- `openai` - OpenAI provider support
- `google` - Google Gemini provider support
- `httpx` - HTTP client
- `xml` - XML parsing (will be replaced by txml)
- `fal` - Fal.ai integration
- `plot` - Visualization tools

### Development Groups
- `dev` - Testing and linting tools
- `doc` - Documentation generation
- `msgtrace` - Observability service development

## Development Workflow

### Code Quality
- **Linter**: Ruff
- **Formatter**: Ruff
- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml`
- **Testing**: pytest with async support

### Branch Strategy
- Working on `init` branch (fork)
- Will merge improvements back to upstream `main` when ready
- Focus on stabilizing core before merging documentation and examples

## Key Architectural Decisions

### 1. Modular Observability
Moving from built-in OpenTelemetry to external msgtrace service allows:
- Dedicated trace visualization and analysis
- Separation of concerns
- Better maintainability

### 2. Rust-Based Parsers
Transitioning from Python XML parsers to Rust implementations (txml, tsn):
- Performance improvements
- Type safety at parser level
- Shared implementation across projects

### 3. Environment Management
Using msgspec-ext instead of custom implementation:
- Battle-tested dotenv handling
- Superior type validation
- Reduced maintenance burden

### 4. Model Provider Architecture
Standardized provider interface in `src/msgflux/models/providers/`:
- Consistent API across providers
- Easy to add new providers
- Provider-specific optimizations

## MCP Integration

The project includes Model Context Protocol (MCP) support in `src/msgflux/protocols/mcp/`:
- Authentication guide available
- Integration summary documented
- Improvements tracked in markdown files

## Testing Strategy

- Comprehensive test suite in `tests/`
- Async test support with pytest-asyncio
- Coverage tracking with pytest-cov
- Mock support for external services

## CI/CD

Excellent CI/CD setup in `.github/workflows/`:
- `ci.yml` - Continuous integration
- `publish.yml` - Package publishing
- Dependabot configuration for dependency updates

This CI/CD work will be migrated to the upstream repository.

## Documentation

Documentation uses MkDocs with Material theme:
- API documentation with mkdocstrings
- Markdown inclusion support
- Currently in .gitignore during restructuring

Interactive documentation available in Colab:
https://colab.research.google.com/drive/15jTAvYlhlg4vd4YErfac9vtHDp4ozIz9?usp=sharing

## Next Steps

1. **Immediate**: Review and commit current changes
2. **Short-term**:
   - Integrate txml/tsn references
   - Complete msgtrace-sdk integration
   - Migrate CI/CD to upstream
3. **Medium-term**:
   - Replace OpenTelemetry with msgtrace
   - Update documentation
   - Commit examples
4. **Long-term**:
   - Merge improvements to upstream
   - Release stable version

## Notes for AI Assistants

- Always check for Rust-based parser implementations (txml, tsn) before modifying Python parsers
- Be aware that telemetry code will be replaced - don't invest heavily in refactoring it
- The project is in transition - some code references old patterns
- CI/CD patterns from this fork should be preserved when working on workflows
- examples/, docs/, and next_version/ are ignored - don't try to commit them yet

## Contact

- **Author**: Vilson Rodrigues
- **Email**: vilson@msgflux.com
- **Homepage**: https://github.com/msgflux/msgflux
