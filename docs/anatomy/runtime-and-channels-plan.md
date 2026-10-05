# Runtime compartilhado e channels

Status: proposta para discussão; nenhuma implementação autorizada por este documento.
Data: 2026-10-05.

## Objetivo

Um runtime pode manter várias conversas e executar agentes acessíveis pela TUI,
HTTP e, posteriormente, canais sociais. A conexão que observa uma execução não
é sua proprietária. Reusar checkpoints, AgentInbox, approvals, TaskStore,
AgentWorkspace e recuperação existentes.

O backend local começa com uma API pequena e HTTP/SSE. Channels são adapters
sobre essa API, inclusive quando usados dentro do mesmo processo. Nenhum
adapter implementa outra máquina de execução, scheduler ou armazenamento do
histórico do Agent.

## MVPs encontrados

| Branch remota | Commit consultado | O que já entrega |
| --- | --- | --- |
| `feat/channels-http-stream` | `12dfbc0ffb7fd91e73a48e857a5cefc03039fe81` | Registry, pre/post, lifecycle hooks, auth, limites, FastAPI e streaming Chat Completions |
| `feat/social-boundary-slack` | `7f941bcbf7e8cdc2ae486b95ac66ff9a2f7703cd` | Boundary social, Telegram e Slack; webhooks, comandos, roteamento e attachments |
| `feat/social-boundary-discord` | `939d38024d41ba7603a7b5753f0c88e14aec0d96` | Boundary anterior mais Discord Interactions, confirmação e respostas posteriores |

Leitura feita com `git show`, sem checkout nem alterações nessas branches.
Não foi encontrado um adapter de WhatsApp nas três árvores consultadas; isso
não afirma que inexista em outras refs.

Fontes remotas:
[HTTP](https://github.com/msgflux/msgflux/tree/12dfbc0ffb7fd91e73a48e857a5cefc03039fe81/src/msgflux/channels),
[Slack](https://github.com/msgflux/msgflux/tree/7f941bcbf7e8cdc2ae486b95ac66ff9a2f7703cd/src/msgflux/channels),
[Discord](https://github.com/msgflux/msgflux/tree/939d38024d41ba7603a7b5753f0c88e14aec0d96/src/msgflux/channels).

### Reaproveitar

- Separação entre protocolo externo, registry, processamento e Agent.
- Registro por nome, metadados de agentes expostos e comandos personalizados.
- Autenticação/verificação antes da execução; autorização da rota escolhida.
- Normalização de mensagens e attachments por adapter.
- Processadores de entrada e saída e suporte de streaming específico do protocolo.
- Tests de payloads, webhooks, envio, comandos e limites como referência.

### Redesenhar para a base atual

Nos MVPs, HTTP chama `agent.acall()` diretamente. A boundary social também chama
Agent diretamente, guarda tasks ativas em dictionaries e usa bus/dedupe em
memória. O caminho social consultado pede `stream=False`. Isso limita retomada
após reiniciar e não constitui um registro durável de admissão ou entrega.

O novo fluxo deve ser `adapter → serviço → execução existente → observação →
adapter`. Portar partes pequenas e revisadas; não fazer cherry-pick integral de
branches antigas com contratos de Agent anteriores aos atuais.

## 1. Identidades e isolamento

| Identidade | Significado |
| --- | --- |
| `agent_id` | Definição pública/configuração/factory do agente, como `main` ou `support` |
| `thread_id` | Conversa canônica, independente de transporte |
| `run_id` | Uma execução da conversa |
| `request_id` | Identificador usado na admissão e dedupe de um pedido |
| `channel_id` | Adapter/instalação que transportou a mensagem |
| `external_conversation_id` | Destino/conversa da plataforma externa |
| `principal` | Identidade autenticada e sua autoridade |

Um agente público pode atender muitas threads. Compartilhar sua definição não
significa compartilhar um objeto Python mutável entre execuções sem controle.
A factory e a propriedade dos recursos precisam permitir isolamento por thread.

A conversa externa é resolvida por uma binding persistente, incluindo instalação,
conta/tenant, agente e a chave de conversa definida pelo adapter. IDs brutos de
mensagem, usuário ou sala não bastam como chave global. Slack pode usar thread;
Discord Interactions do MVP usa sala e usuário. Documentar cada política.

Por padrão, conversas de canais diferentes criam threads diferentes. Compartilhar
uma thread entre TUI e Slack exige uma associação explícita e autorizada. O
mesmo thread_id não concede acesso. A seleção de conversa fica no frontend,
sem alterar uma sessão global selecionada no servidor.

## 2. Serviço de runtime

Criar primeiro um serviço Python independente de Textual, FastAPI ou SDKs sociais.
Nomes propostos, ainda sujeitos ao review: `AgentService` e `AgentServiceClient`.
O uso embutido e o cliente HTTP implementam as mesmas operações públicas.

Contrato inicial:

- Listar agentes públicos e threads autorizadas.
- Abrir uma thread com configuração e workspace definidos pelo host.
- `submit`: admitir entrada e retornar receipt com request_id/thread_id/run_id.
- `steer`: admitir mensagem no AgentInbox da execução alvo.
- `snapshot` / `watch`: observar estado e atualizações.
- `interrupt`: cancelamento explícito da execução alvo.
- Responder a approvals pelo mecanismo atual, com identidade/revisão esperadas.
- Inspecionar e solicitar a retomada do run pelo caminho de recuperação atual.

Records de fronteira usam `msgspec.Struct`, versionamento explícito e payloads
serializáveis. Referências ao Agent, workspace, locks, credentials ou handlers
não atravessam a API. Não transmitir kwargs arbitrários do cliente para Agent.

O serviço mantém execuções fortemente referenciadas até seu encerramento.
Fechar watcher ou conexão HTTP não cancela a execução. Uma execução principal
por thread; threads independentes podem executar em paralelo. Estado de
ocupação precisa ser decidido pelo serviço, não por cada frontend.

### Admissão e recuperação

Persistir um receipt mínimo antes de confirmar aceitação ao cliente:
identidade e escopo do pedido, hash do conteúdo, input normalizado, destino,
run_id e estado de admissão. Repetir o pedido retorna a admissão existente;
reusar a chave com conteúdo diferente causa conflito.

O cliente nativo deve gerar e preservar request_id nas tentativas do mesmo
pedido. No channel compatível, um header explícito de idempotência pode fornecer
essa chave. IDs de tracing/correlation não viram chaves de dedupe automaticamente;
sem uma chave estável do caller, não prometer dedupe de retries arbitrários.

AgentInbox dedupe não substitui esse receipt: o inbox está ligado a um run e
seus registros podem ser consumidos. Também não usar TTL em memória como prova
de admissão durável.

A admissão é metadado do serviço, não outro TaskStore. Antes de implementar seu
store, conferir se a criação condicional de um run/checkpoint existente pode
atender todo o contrato sem alterar o lifecycle. Se não puder, adicionar um
pequeno journal SQLite do serviço. Não prometer transação global entre stores
independentes: testar recuperação em cada fronteira de escrita.

Após crash, uma admissão aceita mas ainda não iniciada pode ser despachada;
um run iniciado precisa da inspeção/reconciliação atual. Não reenviar seu input
como run novo nem repetir efeitos incertos. Lease expirada não prova parada do
worker antigo. Inicialmente, runs interrompidos cuja retomada exige decisão do
host aparecem para retomada explícita.

## 3. API nativa local e SSE

Proposta de superfície, com base `/api/v1`:

| Operação | Endpoint proposto |
| --- | --- |
| Health e versão | `GET /health` |
| Agentes expostos | `GET /agents` |
| Listar/criar threads | `GET/POST /threads` |
| Snapshot | `GET /threads/{id}` |
| Admitir input | `POST /threads/{id}/inputs` |
| Steering | `POST /threads/{id}/runs/{run_id}/inbox` |
| Observar | `GET /threads/{id}/watch` |
| Interromper | `POST /threads/{id}/runs/{run_id}/interrupt` |
| Retomar | `POST /threads/{id}/runs/{run_id}/resume` |
| Approval | `POST /threads/{id}/approvals/{approval_id}/decision` |

`watch` entrega um snapshot inicial e depois eventos. O snapshot para apresentação
traz user messages, commentary e final answers, estado das execuções e approvals.
Cards históricos de tools são opcionais. O progresso vivo pode continuar volátil;
reiniciar o processo recompõe a interface pelo estado durável disponível.

Capturar snapshot e assinatura sem janela que perca eventos, reutilizando o
contrato do watcher atual. Identificar itens/execuções para que um terminal já
presente no histórico e seu evento posterior não apareçam duas vezes. Não
atribuir atomicidade global ao conjunto de stores ao produzir o snapshot.

SSE usa frames tipados/versionados e heartbeat. No primeiro contrato, reconectar
significa obter um snapshot novo. Não usar `Last-Event-ID` como promessa de replay
persistente: ExecutionEvent não tem um ordinal durável completo. O feed de
checkpoints com cursor continua distinto. Overflow desconecta o observador e
permite nova assinatura; não cancela o produtor.

Usar Litestar com um servidor ASGI, por exemplo Uvicorn, como extra opcional
e imports lazy. Reaproveitar dos MVPs os contratos e formatters independentes
de FastAPI. O núcleo e a fronteira HTTP usam msgspec.Struct; não duplicar esses
schemas em modelos Pydantic. Um único app HTTP pode montar a API nativa e,
depois, a API de compatibilidade e webhooks.
Credenciais do provider ficam no backend. O endpoint local é autenticado e usa
loopback por padrão; a autoridade do workspace vem do host.

### Tecnologia e codecs: revisão de 2026-10-05

Recomendação: Litestar, pela integração com msgspec.Struct em validação,
serialização e geração de OpenAPI. Pydantic não é uma dependência obrigatória
do pacote consultado. Litestar também traz dependências próprias; não assumir
que sua instalação completa será menor sem medir a árvore incremental.

FastAPI continua tecnicamente viável: uma Response com bytes permite usar
msgspec para JSON ou MessagePack, com parsing e documentação explícitos nos
pontos necessários. Isso exige mais adaptação para nossos schemas do que o
caminho validado com Litestar.

Validação isolada, sem mudar dependências do projeto: Litestar 2.24.0 recebeu e
retornou msgspec.Struct em JSON, rejeitou tipos inválidos e campos desconhecidos,
recebeu/retornou MessagePack, gerou OpenAPI e produziu SSE tipado. Pydantic não
estava instalado nesse ambiente. Esse teste confirma compatibilidade funcional,
não throughput, consumo de memória nem reconexão do runtime.

Formato inicial: JSON para comandos/snapshots e JSON dentro de SSE para watch.
MessagePack pode ser habilitado na API nativa com negociação explícita de
Content-Type/Accept, usando os mesmos Structs. O suporte do framework não é uma
promessa de negociação automática para todas as rotas. Chat Completions continua
com JSON e seu streaming compatível.

SSE exige texto UTF-8: não enviar MessagePack bruto dentro de data frames.
Um stream binário precisaria de outro contrato de transporte. JSON também é
transmitido como bytes; possíveis ganhos de MessagePack vêm da representação e
do custo de encode/decode. Não assumir ganho perceptível em deltas pequenos;
medir snapshots e operações reais antes de adicionar outro caminho padrão.

Fontes: [tipos customizados](https://docs.litestar.dev/2/usage/custom-types.html),
[MessagePack e SSE](https://docs.litestar.dev/2/usage/responses.html),
[requests MessagePack](https://docs.litestar.dev/2/usage/requests.html),
[Response direta no FastAPI](https://fastapi.tiangolo.com/advanced/custom-response/),
[formato SSE](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream).

## 4. Processo local e propriedade dos recursos

- `vulcano` conecta ao serviço da raiz de estado e inicia um se estiver ausente.
- Um comando explícito de servidor permanece ativo até shutdown solicitado ou
  controle do supervisor. Sintaxe final será definida junto ao entrypoint.
- Um endereço remoto explícito somente conecta; não dispara outro servidor.
- Lock de inicialização, segunda verificação e handshake de readiness/versão
  evitam dupla inicialização. PID ou arquivo de endpoint, isoladamente, não
  provam que o serviço correto está vivo.
- Discovery e estado de controle podem ficar em `~/.msgflux/runtime/`.
  Preservar `config.toml`, `accounts/` e `threads/<thread_id>/` como organização
  existente; não mover checkpoints/approvals para o journal de serviço.
- Fechar o último frontend permite timeout de inatividade configurável somente
  quando não existe trabalho pendente. Não encerrar um run ativo ou uma espera
  por approval porque seu observador saiu.

Modelos, stores e workspaces pertencem ao backend. Agents recebem dependências
por scope; a TUI não fecha recursos compartilhados ao trocar a conversa visível.
`coding/host.py` deixa de manter uma única seleção global como proprietária do
runtime. A configuração de coding monta uma definição de agente; canais de
outros agentes não precisam adotar toda a configuração do Vulcano.

## 5. Channels futuros

Channel é uma adaptação de entrada e saída sobre o serviço. Pode morar no mesmo
processo ou em um gateway que chama o cliente HTTP. O transporte não define
quantas conversas ou Agents existem.

Fluxo de entrada:

```text
verificar/autenticar → normalizar → comando ou pré-processamento
→ resolver destino → autorizar destino final → admitir no serviço
```

Fluxo de saída:

```text
observar resultado/estado → formatar ou pós-processar → entregar ao destino
```

Manter payload bruto e detalhes de SDK na boundary. Contexto do processor pode
conter origem, principal e IDs; isso não vira uma dependência ampla automaticamente
injetada nas tools. Metadata externa não concede permissões nem seleciona um
workspace arbitrário. Reautorizar se o pré-processamento mudar o destino.

### Pré/pós-processamento e comandos

- Pré-processadores normalizam conteúdo e produzem comandos/inputs tipados.
- Processadores de resultado formatam a apresentação; não reescrevem checkpoints
  ou o histórico enviado ao modelo.
- Transformação incremental deve ser distinta de pós-processamento da resposta
  inteira. Exemplo: adicionar um prefixo uma vez exige estado por entrega, não
  substituir strings independentemente em cada token.
- Processadores de fronteira devem ser puros quando possível; efeitos externos
  personalizados precisam de contrato de idempotência próprio. Dedupe de um
  pedido não transforma um callback arbitrário em exactly-once.
- Comandos de domínio, como alterar modelo ou interromper run, chamam o serviço.
  `/copy` e `/quit` continuam comandos da interface.
- Reusar o registro de comandos das CodingExtensions como referência; não enviar
  closures ou widgets ao servidor. Uma extensão pode declarar contribuições
  para backend e frontend separadamente, sem imports cruzados obrigatórios.

### Capacidades de entrega

SSE suporta deltas. Plataformas sociais podem permitir editar uma mensagem,
enviar atualizações agrupadas ou somente a resposta final. O adapter declara
suas capacidades; não mandar um post por token por padrão. Limites de formato,
attachments e janelas de resposta pertencem ao adapter.

Para um webhook que confirma admissão antes da resposta, a entrada precisa estar
salva antes do ACK de aceitação. Validar a janela real da plataforma ao escrever
cada adapter. Handshakes de verificação sem trabalho de Agent são outro caminho.

Entregar uma resposta social após restart exige uma outbox persistente, separada
da execução: referência ao resultado, destino, payload de apresentação, estado
e tentativas. Nunca reexecutar Agent porque `send()` falhou. Tokens de callback
não devem ser armazenados indiscriminadamente como metadata; resolver credenciais
pela instalação e tratar sua validade no adapter. Se a plataforma não oferecer
idempotência/consulta de entrega, manter a possibilidade de entrega incerta.
Não prometer exactly-once na publicação externa.

### Chat Completions como channel

Preservar `/v1/chat/completions`, onde `model` escolhe um agente publicado.
Esse adapter traduz para comandos do serviço e formata respostas compatíveis.
Não misturar snapshots, approvals ou tool events internos com chunks dos SDKs.
Tools executadas internamente não são automaticamente tool calls delegadas ao
cliente que chama o agente como modelo.

Contrato padrão proposto: `messages` representa o contexto fornecido pelo caller
para uma execução isolada, sem anexar todo esse histórico a uma thread existente.
Repetir o mesmo request_id recupera a mesma admissão. Uma vinculação a thread
persistente é uma extensão explícita e precisará de regras para reconciliação
de histórico; não inferir identidade pelo conteúdo do prompt.

Começar com resposta final e streaming de conteúdo final. Commentary só deve
entrar quando houver um campo/extensão negociada: não concatená-lo à final answer
silenciosamente. Backend continua executando se o consumidor do stream sumir;
documentar esse contrato e oferecer interrupção explícita via API nativa.

## 6. Follow-up e steering

Steering usa AgentInbox e é drenado na fronteira segura antes da chamada seguinte
ao modelo, após a rodada de tools. A interface mostra a admissão/estado da mensagem.

Follow-up da TUI permanece uma fila da interface, despachada após `run.end`.
Um futuro canal social que confirmar recebimento de um follow-up não pode depender
de uma fila volátil em uma TUI ausente: sua fila precisa ser durável no adapter
ou na boundary de canais. Isso não altera automaticamente o contrato escolhido
para a TUI nem requer mover follow-ups ao loop do Agent.

## 7. Entrega incremental e arquivos

As primeiras quatro etapas preparam a base local. As demais são futuras.
Não misturar a migração dos MVPs sociais com a refatoração de propriedade do runtime.

### PR 1 — serviço embutido e execução independente

Arquivos propostos: `runtime/service.py`, `runtime/service_types.py`,
`coding/session.py`, `coding/cli.py`, tests novos de serviço e sessão.
Inspecionar o lifecycle e adapters de CheckpointStore antes de escolher a forma
mínima de admissão persistente. Se precisar de journal próprio, adicionar um
módulo pequeno e testes de recuperação, sem duplicar TaskStore.

Entregar submit/receipt, snapshot/watch, interrupt e ownership por thread.
Manter factory de dependências explícita. Sem sockets ou dependência web.

### PR 2 — TUI como observadora e integração do inbox

Arquivos: `coding/host.py`, `coding/session.py`, `coding/tui/app.py`,
`coding/extensions/`, testes de TUI/session/host.

Trocar o acoplamento atual a stream_events por submit + watch. A implementação
interna do serviço pode consumir stream_events para publicar eventos; sua vida
não depende do watcher. Conectar steering, fila de follow-up da TUI e reconstrução
de user/commentary/final. Não incluir painel novo de TaskStore.

### PR 3 — API nativa HTTP/SSE e cliente

Arquivos propostos: `channels/http/app.py`, `channels/http/schemas.py`,
`channels/http/client.py`, `channels/__init__.py`, testes de HTTP/SSE,
`pyproject.toml` e `uv.lock` pelo fluxo uv.

Expor o contrato do PR 1, autenticação local, snapshot + updates e comandos.
Reusar cliente HTTP e utilitários SSE onde aplicáveis. Serializer explícito para
snapshot/eventos; evitar vazar objetos vivos ou provider_state opaco à interface.
Introduzir primeiro o servidor explícito e seleção de cliente por endereço.

### PR 4 — inicialização automática e reconexão local

Arquivos propostos: `channels/cli.py`, módulo pequeno de descoberta/lifecycle,
`coding/cli.py`, `coding/config.py`, tests independentes de processo.

Discovery, lock, readiness, attach, shutdown e idle policy. Tornar autostart o
fluxo normal da TUI somente após validar reconnect e trabalho sobrevivendo ao
fechamento do frontend. Importar diretamente a API do serviço continua válido
para embedding e testes.

### PR 5 futuro — compatibilidade HTTP e extensão de channels

Portar schemas/formatters selecionados de `channels/http/openai.py` e registry
experimental, agora ligados ao serviço. Definir registro de processors e
comandos sem acoplamento a Textual. Testar usando cliente SDK e fixtures locais,
sem exigir chamadas reais a modelos.

### PR 6 futuro — primeiro canal social

Escolher um adapter existente, por exemplo Slack. Portar normalização/verificação
e entrega; implementar binding de conversa, admissão durável e outbox.
Só depois expandir para Discord/Telegram/WhatsApp. Não presumir que o MVP Discord
Interactions já seja um bot para mensagens livres via Gateway.

## 8. Riscos e testes obrigatórios

| Risco | Verificação |
| --- | --- |
| Desconexão aborta Agent | Desconectar watcher/HTTP/TUI; run e tool continuam; novo cliente recebe snapshot |
| Pedido duplicado | Repetir request_id antes/depois de crash; mesmo receipt/run; payload divergente gera conflito |
| ACK sem admissão | Matar processo entre persistir receipt, confirmar aceitação, despachar e salvar checkpoint |
| Corrida na thread | Dois clientes simultâneos: uma execução; steering e busy policy determinados pelo serviço |
| Snapshot perde ou duplica terminal | Attach durante deltas/commit; reducer por IDs e estado final coerente |
| Stores com commits separados | Crash nas fronteiras; reconciliar receipt/checkpoint sem reexecutar efeito incerto |
| Bootstrap concorrente | Duas TUIs lançam: uma instância; arquivo stale/PID reciclado/versão incompatível |
| Consumidor lento | Limite de fila, desconexão e reassinatura; produtor segue saudável |
| Escopo cruzado | Threads/workspaces diferentes em paralelo; acesso/approval não autorizado é rejeitado |
| Canal sem token streaming | Saída final ou edições agrupadas, sem flood e com destinos corretos |
| Falha de publicação | Outbox retoma entrega sem chamar modelo/tool outra vez; resultado externo incerto permanece explícito |
| HTTP compatível incorreto | SDK lê chunks/finish/[DONE]; commentary e tool activity não contaminam final answer |

Reusar testes de durabilidade, processos, approvals e receipts existentes.
Aplicar o durability gate quando checkpoints/recuperação forem afetados.
Os PRs iniciais podem usar modelos falsos; uma integração real pequena confirma
que a mudança de ownership não altera streaming nem tools.

## 9. Documentação prevista

- Atualizar `docs/learn/coding.md` com submit/watch, reconnect e fila da TUI.
- Atualizar `docs/learn/nn/agent/event-streaming.md` com execução versus observação
  e fronteiras entre SSE visual e feed persistente de checkpoints.
- Introduzir `docs/learn/channels/index.md` e página HTTP para API nativa,
  embedding, server explícito e autostart, com exemplos completos.
- Adicionar exemplos de processors/comandos e compatibilidade no PR 5.
- Publicar docs específicas de cada plataforma junto com seu adapter.
- Atualizar navegação e validar MkDocs; não publicar docs dos MVPs antigos como
  se as APIs já estivessem disponíveis na main.

## Decisões para review

Proposta usa conversas separadas por canal por padrão, vínculo explícito para
compartilhar uma thread, HTTP/SSE como primeiro transporte, backend iniciado sob
demanda, autoridade do workspace no host e follow-up da TUI na interface.
São escolhas revisáveis. A implementação começa somente após discutir e aprovar
este plano e os contratos de admissão; nenhum código de runtime foi alterado.
