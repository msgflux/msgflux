# Revisão de pi-durable

Data: 2026-10-05. Revisão estática do código, documentação e testes.
Nenhum teste do Pi foi executado; não houve alteração no runtime do msgflux.

## Referências analisadas

- Pi: `origin/main`, commit `5b6c792b424e73edefbfa558b901bcd64788dad2`.
  O manifesto identifica `@earendil-works/pi-durable` como versão 1.0.3.
- msgflux: `feat/coding-resume`, commit
  `2db6943f2917b2d3aa2ef9509a9938556bb64e2e`, após rebase sobre o PR #206.
- Pesquisa coordenada, com contribuições verificadas de dois subagentes Luna
  e leitura direta pelo coordenador de recuperação, scheduler e testes. Um
  terceiro subagente encontrou falha de sandbox e não forneceu evidências.

O pacote declara API experimental. A integração de coding analisada está em
`packages/coding-agent/src/experimental/durable/`; suas garantias não devem ser
atribuídas automaticamente à CLI principal do Pi.

Fontes: [pacote](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/package.json),
[estado experimental](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/README.md#L1),
[integração de coding](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/coding-agent/src/experimental/durable/README.md).

## 1. O desenho central

`Session` oferece uma linha de commits para entradas imutáveis da conversa,
tarefas, submissões e documentos JSON. Uma transação pode alterar essas entidades
juntas. Os documentos guardam configuração, fila, custos, identidade do provider
e estado de apresentação. Efeitos externos ficam fora da transação.

O harness resolve extensões pelo registry vivo; o estado salvo guarda nomes e
escolhas. Isso permite reconstruir dependências após reiniciar, sem persistir
funções, processos ou conexões. A resolução por fases também define quando uma
mudança de implementação passa a valer.

Fonte: [contrato de Session](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/docs/spec.md#L41),
[registry](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/registry.ts).

No msgflux os stores de checkpoints, tasks, inbox e approvals têm contratos
próprios. A atomicidade do checkpoint inclui estado e transição naquele store;
isso não significa uma transação global entre todos os stores. A coordenação de
recuperação já trata essa separação. Não há motivo demonstrado nesta revisão
para substituir esses contratos por uma Session universal.

## 2. O principal insight: apresentação reconstruível

O documento `pi.live` guarda a geração parcial, ferramentas em andamento,
progresso, retries e compactions. As parciais são agrupadas em gravações com
intervalo padrão de 100 ms e apenas uma gravação em andamento. A apresentação
só observa dados após o commit. Dados ainda pendentes podem se perder num crash;
100 ms é a cadência configurada, não uma garantia universal de perda máxima
quando o armazenamento está lento.

Fonte: [LiveDoc](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/live.ts#L43),
[gravação das parciais](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/generation.ts#L349).

A TUI lê uma projeção da conversa e dos documentos. Retomar a interface não
exige reproduzir todos os callbacks que montaram seus widgets. A mesma projeção
expõe fila, modelo, usage e tarefas. Os eventos de apresentação são derivados
dos commits.

Fonte: [view](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/view.ts),
[eventos](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/events.ts#L137).

Nosso `EventHub` oferece histórico durável mais projeções do processo atual;
o feed `observe_checkpoints` é outra API, persistente e com cursor. Parciais do
modelo são consolidadas em itens coerentes nas fronteiras de execução. Portanto,
a diferença útil é persistir uma apresentação parcial opcional, sem confundir
esse registro com mensagens prontas para replay ao provider.

Fontes locais: `src/msgflux/runtime/event_hub.py`,
`src/msgflux/data/stores/observation.py`,
`docs/anatomy/checkpoints-and-replay.md`.

A integração experimental acrescenta uma fila no controller para aplicar ações
na ordem de admissão. O cancelamento fica fora dessa fila, para não esperar uma
operação lenta. Isso é uma simplificação útil para a concorrência da TUI, sem
exigir um novo protocolo RPC.

Fonte: [controller](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/coding-agent/src/experimental/durable/runtime.ts#L224).

## 3. Consumidores lentos: recuperar o estado atual

Os watches do Pi limitam a fila a 100 frames. Quando um consumidor fica para
trás, os frames pendentes viram um snapshot completo recente. Reconectar começa
na visão atual, sem replay de todos os eventos anteriores.

Fonte: [watch e limite de fila](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/session/observation.ts#L152).

Esse comportamento serve para convergência visual. Nosso feed de checkpoints
já oferece paginação, cursor e detecção de lacunas; deve conservar esse contrato.
Para a TUI, uma opção de ressincronização por snapshot pode ser mais adequada
que encerrar a observação em overflow. Não devemos silenciosamente descartar
histórico em consumidores que precisam de todas as transições.

## 4. Recuperação não significa repetir qualquer ferramenta

O Pi grava a intenção antes do efeito. Na recuperação, uma tool é repetida
somente quando a política salva e a implementação atual declaram `replay="safe"`.
O padrão é `unsafe`: a chamada recebe um resultado de interrupção, com saída
parcial persistida e aviso de que pode ter executado parcialmente. A geração
pode então continuar. Isso não prova o desfecho do efeito externo.

Fonte: [ToolTask e replay](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/tool.ts#L45),
[testes de recuperação](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/test/harness-tools-recovery.test.ts#L109).

Reabrir não continua a mesma conexão SSE nem implica reconectar aos pipes do
mesmo bash. Uma execução marcada safe ainda pode repetir efeitos se a declaração
estiver errada. A recuperação usa o ambiente atual; um teste confirma que o cwd
pode ter mudado. Esse contrato não substitui nossos command receipts,
reconciliação de efeitos incertos, verificação de identidade do workspace e
revalidação da autoridade atual.

## 5. Submissões e tarefas como API de interação

Uma entrada tem identidade e estado observável. `requestId` permite reencontrar
a submissão após reconexão; cancelar a espera não cancela o trabalho. Durante
uma conversa ocupada há políticas explícitas para steering, follow-up e rejeição.
O dedupe consultado é por conversa/requestId e tipo: ele não verifica igualdade
do conteúdo de uma repetição. Não copiar esse detalhe sem definir nosso contrato.

Fonte: [admissão das submissões](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/submissions.ts#L144),
[testes de dedupe](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/test/harness-submissions.test.ts#L88).

`taskGraph()` expõe ownership, espera por outras tarefas, background e intenção
de cancelamento, sem incluir os grandes payloads dos checkpoints. Isso dá uma
boa referência para uma barra lateral de runs/tasks e para inspecionar
subagentes. O msgflux já tem tasks e AgentInbox; integrar esses recursos à
CodingSession evita construir outra fila na TUI.

Fonte: [grafo das tarefas](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/task-graph.ts#L17).

## 6. Ambiente, forks e identidade do provider

`ExecutionEnv` reúne filesystem e execução. Sua identidade permite serializar
mutações de arquivo entre objetos que representam o mesmo ambiente. Essa direção
é coerente com nosso AgentWorkspace. O Pi passa APIs amplas e Context para as
operações; devemos manter nossa injeção explícita de dependências.

Fonte: [ExecutionEnv](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/env/index.ts#L171).

Documentos têm políticas de fork `asOf`, `current` e `initial`; configurações
podem acompanhar o ponto histórico enquanto progresso vivo começa vazio.
A identidade do provider é persistida por conversa, permanece em retries e
compaction e começa nova no fork/child. Nosso Codex já usa o thread_id para
cache affinity. Vale revisar a granularidade thread/namespace nas futuras
interfaces de fork e subagentes, sem declarar uma falha de cache existente.

Fontes: [forks](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/session/forks.ts#L25),
[provider identity](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/src/harness/provider.ts#L6).

## 7. Limites que importam

- O storage do pacote pressupõe um processo proprietário; não tem exclusão
  entre processos no próprio contrato. A TUI experimental acrescenta um lock.
  Não transferir essa premissa para nossos workers e leases.
- SQLite usa WAL com synchronous NORMAL; JSONL oferece fsync opcional.
  Persistência não implica que o último commit sobreviva a toda falha de energia.
- Falha de storage com resultado incerto invalida a Session aberta. Publicação
  e retry não devem continuar como se o commit tivesse sido rejeitado com certeza.
- Guards são extensões selecionáveis. Nossa autoridade do workspace deve continuar
  independente da seleção de tools/extensões feita para o modelo.
- Watches são projeções de estado, não um journal completo de auditoria.

Fontes: [storage e propriedade](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/README.md#L535),
[invariantes de falha](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/docs/spec.md#L41),
[limites e não objetivos](https://github.com/earendil-works/pi/blob/5b6c792b424e73edefbfa558b901bcd64788dad2/packages/durable/docs/spec.md#L4630).

## Prioridade sugerida para msgflux

1. Tornar a TUI reconstruível a partir de snapshot e estado atual das tasks,
   approvals e fila. Preservar a distinção entre observar e executar.
2. Integrar AgentInbox à CodingSession com steering/follow-up explícitos e fila
   visível. Definir dedupe/admissão antes de introduzir handles de submissão.
3. Avaliar persistência opcional e agrupada do progresso visual. Medir custo e
   definir fronteiras/versões; não gravar checkpoints completos por token.
4. Expor uma projeção pequena de runs/tasks para a barra lateral e subagentes.
5. Usar os testes de falhas do Pi como referência adicional para conformance,
   mantendo nossos testes independentes de processo, CAS e reconciliação.

São direções de estudo, não um plano aprovado de implementação. Preservar a
base existente de workspace, permissions, approvals e recuperação; a revisão
não justifica troca de runtime, adoção de Chord ou novos backends para a v1.
