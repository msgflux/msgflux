# Plano: Workspace como dependência explícita das ferramentas

Status: API de runtime, migração de builtins e documentação implementadas.
A adoção na branch experimental da TUI fica em uma entrega dependente.
Branch: `feat/workspace-api`, baseada em `upstream/main` (`8c225b3f`).

## Objetivo

Oferecer uma API pública pequena que reúna execução, leitura e alteração de
arquivos. As tools builtin e as tools customizadas devem declarar a mesma
dependência `workspace` e chamar seus métodos diretamente.

O alias para `ExecutionEnvironment` da tentativa anterior é insuficiente:
o usuário ainda precisa conhecer filesystem, executor, editor, escopo e grants.
A nova API deve encapsular essa composição, reutilizando esses componentes.
Bubblewrap, novos backends, SSH e configuração de sandbox ficam fora deste plano.
O trabalho experimental anterior permanece na worktree separada.

## API proposta

```python
from msgflux.nn import Agent
from msgflux.runtime import AgentWorkspace
from msgflux.tools import Hidden, tool_config


@tool_config(runtime_inputs=["workspace"])
async def inspect_project(*, workspace: Hidden[AgentWorkspace]) -> str:
    """Read the project description and list tracked files."""
    description = await workspace.aread_text("README.md")
    result = await workspace.arun(["git", "ls-files"], timeout=10)
    return description + "\n" + result.stdout.decode("utf-8", errors="replace")


agent = Agent(
    name="main",
    model="openai/gpt-6-luna",
    workspace=AgentWorkspace.local("."),
    tools=[inspect_project],
)
print(agent("Inspecione o projeto."))
```

Este exemplo descreve a API pública implementada. O host configura o
workspace uma vez; a tool declara somente a dependência de que necessita.
Não há parâmetro genérico `ctx` e o modelo não recebe `workspace` no schema.
`Hidden` e `runtime_inputs` continuam seguindo o padrão de injeção existente.
O uso local não exige `with`, `async with`, abertura ou fechamento manual.

Também é possível fornecer o workspace por execução, usando o scope:

```python
from msgflux.runtime import ExecutionScope

scope = ExecutionScope(workspace=AgentWorkspace.local(".", read_only=True))
result = agent("Revise os arquivos sem alterar nada.", scope=scope)
```

O `workspace` do scope seleciona o ambiente dessa execução; o valor do init é
um default. Selecionar outro workspace não concede automaticamente novas
permissões quando já existe autoridade explícita ou herdada. Nenhum dos dois
caminhos requer um gerenciador de contexto no código do usuário.

### Métodos iniciais

Todos os métodos de I/O possuem versão síncrona e versão async com prefixo `a`,
preservando a convenção atual do projeto.

| Métodos | Contrato |
| --- | --- |
| `run` / `arun` | Recebem string de comando Bash ou sequência de argv; convertem internamente para `ProcessRequest`. Retornam o `ProcessResult` existente. |
| `read_text` / `aread_text` | Leitura UTF-8 com limite de bytes explícito e default finito. |
| `read_bytes` / `aread_bytes` | Leitura binária limitada, usada também para imagens. |
| `read_lines` / `aread_lines` | Leitura paginada com offset, limite de linhas e de bytes. |
| `listdir` / `alistdir` | Nomes dos filhos de um diretório. |
| `scandir` / `ascandir` | Entradas estruturadas com limite de quantidade; reutilizam `WorkspaceEntry`. |
| `write_text` / `awrite_text` | Criam ou sobrescrevem arquivo pelo fluxo de edição existente. |
| `edit_text` / `aedit_text` | Substituição exata conforme a semântica atual de `EditTool`. |
| `delete` / `adelete` | Remoção de arquivo pelo fluxo existente de comparação e aprovação. |
| `mkdir` / `amkdir` | Criação de diretório com verificação de permissão. |
| `resolve` | Resolução centralizada de caminhos relativos ao cwd do workspace. |

Na API de execução, uma sequência representa argv, nunca lote de comandos.
`BashTool` continua aceitando seu lote atual e executa cada comando preservando
seus limites combinados. `timeout`, limite de saída e callback de streaming
são encaminhados aos componentes existentes. Exit code diferente de zero
permanece resultado normal; cancelamento e limites mantêm seus erros tipados.

O parser de patch permanece na camada de ferramentas. `ApplyPatchTool` prepara
uma transformação e a aplica através do Workspace; não introduzimos dependência
de protocolos de patch em backends de filesystem.

### Caminhos e capacidades

O Workspace mantém um cwd e centraliza a resolução hoje repetida em `_tool_path`.
Uma vista com outro cwd, quando necessária para tools existentes, compartilha
ambiente e autoridade; não abre outro backend nem aumenta permissões.

`AgentWorkspace.local(root)` oferece caminhos virtuais relativos ao projeto para
operações de arquivo. Execução local conserva a autoridade normal do host:
a raiz de arquivos não constitui isolamento de comandos.

A integração da TUI preserva sua semântica atual de filesystem do host e cwd
no projeto. Essa integração usa o adaptador existente, sem mudar silenciosamente
os caminhos absolutos já aceitos por suas tools.

`read_only=True` retira alteração e execução do workspace local simplificado.
Executar Bash não é tratado como operação somente leitura. Essa informação
fica disponível para o harness selecionar suas ferramentas.

## Composição e injeção

1. Implementar `AgentWorkspace` em módulo novo, por exemplo
   `src/msgflux/runtime/workspace_api.py`, sem renomear o módulo de filesystem.
2. O objeto mantém uma referência ao `ExecutionEnvironment` existente e seus
   componentes privados; filesystem/executor/editor não são necessários na API
   cotidiana. Não expõe conversação, library handle ou serviços sem relação.
3. Adicionar `AgentWorkspace.local(...)` como configuração local utilizável diretamente
   no init ou scope. O factory compõe os adapters locais existentes, política
   local e permissões iniciais do host. Arquivos e processos abrem recursos
   durante a operação e os liberam internamente, inclusive em erros e cancelamento.
   O caso local não mantém conexão que exija fechamento manual pelo usuário.
   Extrair a lógica útil do helper de coding para runtime, sem fazer runtime
   importar o pacote da TUI e sem manter dois helpers equivalentes.
4. Adicionar `workspace=` ao Agent e ao ExecutionScope. O runtime usa o valor
   explícito do scope quando fornecido, ou o default do Agent quando permitido,
   associa esse objeto à execução e injeta a mesma instância nas tools que o
   declararem. Não trocar o ambiente de um run em andamento ou retomado
   silenciosamente; aplicar as verificações de identidade existentes.
5. A API avançada de `ExecutionScope.environment` continua suportada. Quando
   esse caminho for usado, o runtime obtém uma fachada para aquele ambiente,
   consistente durante a execução e suas chamadas de background.
6. A injeção para arquivos e execução usa somente `workspace`, entregando
   `AgentWorkspace`, nunca um alias de `ExecutionEnvironment`.

### Permissões, concorrência e persistência

- O factory local é uma decisão explícita do host sobre acesso local. Ele pode
  fornecer defaults para um Agent sem escopo configurado.
- Um escopo explícito ou herdado com restrições não recebe uma união automática
  com esses defaults. Uma lista vazia explícita de grants continua sem acesso.
- Workspace e environment conflitantes dentro do mesmo scope são erro de
  configuração. Um workspace explícito no scope pode substituir o default do
  init para uma nova execução, respeitando permissões e identidade na retomada.
- Nenhum gerenciador de contexto é obrigatório na API local. Uma eventual API
  explícita de fechamento para conexões persistentes será discutida junto com
  backends futuros; ela não é requisito desta entrega.
- Cwd por execução/tool não pode alterar estado compartilhado entre threads.
  Usar vistas imutáveis para cwd; structs de configuração novas usam msgspec.
- Permissões e cancelamento continuam sendo consultados na execução, não
  armazenados como autoridade na fachada nem serializados em checkpoints.
- O workspace é fornecido pelo host na retomada. Checkpoints não guardam
  objetos de filesystem, executor, conexões nem credenciais.
- Subagentes e jobs em background preservam a associação e as restrições do
  seu escopo. Um workspace recebido não habilita escalada de autoridade.

## Migração obrigatória das tools existentes

| Tool / infraestrutura | Alteração |
| --- | --- |
| `ReadFileTool` | Declara `workspace`; usa leitura de linhas/binários. Continua declarando `handle` para anexar imagens. |
| `BashTool` | Declara `workspace`; usa execução da fachada. Preserva lotes, timeout, background e captura/offload declarados separadamente. |
| `WriteTool`, `EditTool`, `DeleteTool` | Declaram `workspace`; usam operações de edição da fachada. |
| `ApplyPatchTool` | Declara `workspace`; mantém parser e contrato de proposta aprovada. |
| `LsTool`, `GlobTool`, `GrepTool` | Declaram `workspace`; usam listagem e leitura da fachada, preservando limites e algoritmos atuais. |
| `WorkspaceChangeTool` | Prepara e aplica mudanças através do Workspace; deixa de procurar environment global para construir o editor. |
| Guard de aprovação | Resolve o mesmo Workspace usado pela tool e aprova/aplica a proposta exata. |
| Coding host | Configura Workspace uma vez e o associa ao Agent/session; não compõe grants/executor dentro das tools. |

Nomes e schemas públicos apresentados ao modelo permanecem iguais.
Parâmetros Python diretos das tools builtin usam somente `workspace`.
Não há ponte de compatibilidade para filesystem/environment; consumidores
Python diretos e exemplos são migrados nesta entrega.

## Edição e aprovações

As operações de escrita reutilizam WorkspaceEditor e PreparedFileChange.
Não delegam diretamente a métodos de escrita do filesystem que contornem
comparação, limite de bytes ou aprovação.

As tools builtin preservam preparação sem efeitos, diff para revisão e aplicação
da proposta aprovada, mesmo que o arquivo mude entre essas fases.
O código aprovado é reutilizado; não recalcular uma proposta diferente na retomada.

Para tools customizadas, `workspace.write_text` deve funcionar no uso local sem
revisão, mas nunca suprimir uma revisão exigida pelo host. Nesta primeira versão,
se uma custom tool não oferece preparação compatível com o guard e a política
exige revisão, a mutação é recusada explicitamente antes do efeito. O protocolo
existente de preparação permanece disponível para esse caso avançado.
Não expandir este trabalho para execução especulativa de tools ou dry run
arbitrário de funções Python.

## Arquivos e ordem de implementação

### Etapa 1 — Fachada e configuração simples

- Novo `runtime/workspace_api.py` e export em `runtime/__init__.py`.
- `runtime/context.py`: `ExecutionScope(workspace=...)` e associação consistente
  com environment, sem exigir contexto manual do usuário.
- Reutilização de environment, filesystem/editor e helper local de coding.
- `nn/modules/agent/core.py`, `inputs.py` e código de escopo: parâmetro `workspace`
  e regras de conflito/herança. Inspecionar AutoParams, serialização e retomada.
- `nn/modules/tool/extensions.py` e `execution_runtime.py`: injeção explícita.
- Testes da API e do factory com uma custom tool real via Agent/ToolLibrary.

### Etapa 2 — Builtins usando a API

- `tools/builtin/workspace_tools.py`, `workspace_query.py`, `apply_patch.py`.
- `tools/workspace_changes.py` e guards de aprovação que consomem as propostas.
- Integração de captura shell em `nn/extensions/tool_output.py`, se necessária.
- Testes atuais de workspace, shell, imagens, patch, aprovação e background.

### Etapa 3 — Harness e documentação

- Integração na branch da TUI depois de a API de runtime estar verificada.
- `docs/learn/nn/agent/runtime.md`: exemplo curto de configuração e custom tool.
- Documentação existente de builtins e injeção: trocar exemplos e explicar ponte.
- `docs/learn/coding.md` na branch da TUI: configuração local simplificada.

As etapas formam uma entrega focada de API e migração de tools. Integração da TUI,
se precisar de PR separado por ainda estar experimental, depende da entrega de
runtime. Backends novos ficam em outra proposta posterior.

## Verificação e critérios de aceite

- Uma custom tool usa `workspace` sem configurar filesystem/editor/executor/grants
  manualmente para o caso local básico, e sem exigir `with` ou `async with`.
- Configuração no init e por scope funcionam com precedência explícita, sem
  elevação de grants ou alteração silenciosa do workspace na retomada.
- Todas as tools builtin de workspace usam a fachada, e não só novos exemplos.
- Um comando cria arquivo que a leitura da mesma fachada consegue observar.
- Read-only, permissões explícitas, limites de leitura/saída e cancelamento passam.
- Caminhos/cwd, imagens, lotes Bash e schemas de tools mantêm semântica atual.
- Aprovações continuam sem efeitos antes da decisão; proposta aprovada não muda.
- Background, retomada, tool buckets e offload mantêm o workspace correto.
- Tools usam uma única dependência de workspace; recursos conflitantes falham.
- Testes de concorrência verificam que cwd e grants não vazam entre execuções.
- Rodar suíte focada, gate de durabilidade do CONTRIBUTING (pelas mudanças de
  escopo/retomada), suíte offline de CI, Ruff e MkDocs.

## Fora do escopo

Bubblewrap, SSH, outros backends, nova política de sandbox, autenticação, TUI
visual, modelo/provider e mudança no schema das ferramentas enviado ao modelo.


## Resultado da implementação

- `AgentWorkspace.local`, métodos de I/O e execução, e views independentes de cwd.
- Configuração no init de Agent ou por ExecutionScope, com injeção explícita.
- Builtins migrados, preservando schemas e aprovação.
- Nenhum handle de workspace incluído em state_dict/checkpoints.
- Documentação pública atualizada em runtime, builtins e runtime_inputs.
- Ruff e MkDocs passaram; suíte offline final: 3.639 passed / 32 skipped;
  workspace/executores: 314 passed; consumidores/approvals/prompt: 112 passed.
- Sem commits nesta entrega. A integração da TUI permanece na sua branch
  experimental, conforme a separação prevista acima.


## Ajuste: uma única dependência nas tools builtin

A revisão eliminou os parâmetros legados filesystem/environment das assinaturas
builtin e de runtime_inputs. Workspace passa a ser a única dependência para
arquivos e processos; handle e shell_capture mantêm suas funções específicas.
Os adapters internos e scopes avançados continuam utilizando ExecutionEnvironment
para autorização e ciclo de vida. Tools customizadas também usam a fonte workspace; as fontes antigas são removidas. Ordem: assinaturas/helpers, testes de consumidores diretos,
documentação; verificar schemas, cwd, imagens, background e aprovações. Risco:
chamadas Python diretas com os kwargs antigos precisam passar workspace. Nenhuma
mudança nos argumentos que o modelo recebe. Os testes e exemplos serão migrados.


### Cwd pertence ao Workspace

Removidos cwd e self.cwd dos construtores e implementações de todos os builtins.
Uma tool resolve caminhos e executa comandos no cwd do Workspace recebido.
Uma view criada com with_cwd seleciona outro diretório sem mudar o original.
Testes e exemplos configuram o cwd no workspace; schemas do modelo não mudam.


### Nomes públicos

A classe pública é AgentWorkspace; a dependência continua chamada workspace.
Builtin workspace.py foi renomeado para workspace_tools.py. Imports, exports,
consumidores e documentação foram atualizados sem aliases de compatibilidade.
