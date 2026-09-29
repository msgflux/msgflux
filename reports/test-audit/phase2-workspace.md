# Auditoria fase 2: workspace, segurança e persistência

## Escopo e método

Revisão estática de testes na raiz `tests/` ligados a workspace/filesystem, autoridade e permissões, approvals, checkpoints, eventos e tarefas persistentes. Comparei as verificações com os respectivos contratos e caminhos de implementação em `src/msgflux/`. A revisão não executou os testes, não mediu cobertura/mutation score e não alterou código de teste. Casos já tratados no relatório core não foram reabertos como recomendações de remoção.

## Resultado

**Nenhum teste completo é candidato forte à remoção nesta área.** Não encontrei teste sem efeito observável, tautologia, ou repetição integral de outro caso que pudesse ser apagada sem perda aparente. A densidade de testes é alta porque os componentes lidam com isolamento entre escopos, limites de autorização, concorrência, crash/restart e persistência; casos aparentemente próximos exercitam providers ou falhas distintas.

Isso não sustenta uma meta de remover 10%: um percentual escolhido antes de identificar redundância levaria a cortes arbitrários. Esta rodada não encontrou base para esse corte. Uma estimativa percentual confiável exigiria instrumentação de cobertura e/ou mutation testing, além da leitura estática.

## Revisões de baixo risco / simplificação possível

- `tests/test_workspace_display_names.py::test_workspace_display_names` — verifica quatro strings constantes (`Find`, `List`, `Search`, `ApplyPatch`) expostas pelas classes de ferramentas em `src/msgflux/tools/builtin/workspace_query.py`. É um teste simples, mas o nome exibido é consumido pela interface/registro público da ferramenta e cada assert cobre uma classe diferente. **Confiança: alta** quanto à simplicidade; **risco de apagar: baixo, mas não zero** (perda da guarda contra alteração acidental de rótulo). Melhor candidato a consolidar com um teste de registro/descrição caso esse contrato já esteja afirmado lá, após confirmação de cobertura.
- Pares de operações de checkpoint/task/workspace para `InMemory*` e `SQLite*` podem parecer duplicados por terem expectativas parecidas. Eles verificam implementação de providers diferentes; cenários SQLite incluem serialização, reopen, transação, conexão concorrente e comportamento do banco. **Não recomendar remoção** sem uma matriz de conformance que prove a sobreposição exata.

## Áreas verificadas

- Workspace: backend, filesystem local, navegação e query, operações de edição/delete, ferramentas builtin e integrações com executor. Os asserts observam resultados, rejeição de caminhos/recursos, atomicidade, lifecycle, permissões e limites de leitura/busca.
- Segurança e approvals: `test_resource_security.py`, `test_runtime_permissions.py`, `test_approval_store.py` e casos próximos em workspace. As verificações de negação anterior à execução, vinculação da decisão à identidade/recursos e invalidação após mudança correspondem a fronteiras de segurança observáveis.
- Checkpoints: `test_checkpoint_store.py`, `test_checkpoint_revisions.py`, `test_checkpoint_observation.py` e conformance/process tests. Embora alguns cubram revisão ou atomicidade em mais de um nível, os casos de restart, rollback, cursor inválido, concorrência, fork e observação cobrem contratos independentes.
- Eventos: `test_event_buffer.py`, `test_event_hub.py` e `test_event_memory_benchmark.py`. Incluem limites de fila, liberação de referências, publishers em threads, corridas de fechamento, watchers, reconexão e retenção. Os testes do benchmark verificam semântica e inputs, não um limite instável de performance.
- Tarefas: `test_task_store.py`, `test_task_message_queue.py` e `test_task_runtime.py`. As repetições de estado entre memória/SQLite cobrem persistência e concorrência; os testes runtime exercitam injeção, lifecycle de ferramenta/agent e entrega/ack de mensagens.

## Próxima forma de estimar redução

Usar cobertura por linha/branch e mutation testing por pacote para apontar asserts que não detectam mudanças. Em seguida, agrupar apenas testes cujo comportamento mutante e caminho executado sejam equivalentes; sincronismo/async, provider, persistência/reopen, permissões e falhas concorrentes devem permanecer distintos quando mutações relevantes forem detectadas por apenas um deles. Esta auditoria estática não dá suporte a uma estimativa numérica de testes removíveis.
