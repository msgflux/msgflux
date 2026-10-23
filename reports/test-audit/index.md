# Auditoria de utilidade dos testes

## Escopo e método

Esta auditoria identifica candidatos a remoção/refatoração; não remove nem altera testes. A avaliação considera se o teste protege comportamento observável, regressão, contrato, integração entre camadas, tratamento de erro ou compatibilidade. Contagem de `assert`, `pass` em helpers, uso de mocks e skips condicionais não são suficientes, isoladamente, para classificar um teste como inútil.

Cada área foi atribuída a um revisor que inspeciona os testes junto à implementação correspondente. Os resultados detalhados ficam nos relatórios por área e serão consolidados neste índice. Recomendações são conservadoras: "candidato forte" significa que há evidência concreta de ausência de valor próprio; "revisar" aponta sobreposição ou baixa força das verificações, mas requer decisão humana/refatoração; "útil" indica proteção relevante identificada.

## Limites desta revisão

A auditoria é estática e baseada na leitura do código. Não houve remoção, mutação, execução da suíte, medição de cobertura nem experimento de mutation testing. Portanto, não se afirma que um teste seja redundante apenas por parecer semelhante: essa hipótese precisaria de validação removendo-o temporariamente e observando a cobertura comportamental perdida.

Inventário inicial: 215 arquivos `test_*.py` e aproximadamente 3.030 funções `test_*`/`async test_*`. Esses números são contagens textuais da árvore, não contagem oficial do pytest. `pass`, `assert True`, skips ou asserts de não vazio foram usados apenas como pistas para revisão contextual.

## Relatórios por área

- `core.md` — testes de Agent, runtime, workspace e ferramentas na raiz de `tests/`.
- `models.md` — modelos e provedores.
- `specialized.md` — `nn`, dados, protocolos, telemetria, geração, automação e ferramentas.
- `integration.md` — testes de integração, DSL, utilitários e autenticação MCP.

## Próximos passos sugeridos

1. Revisar candidatos fortes e conferir que a implementação não tem contrato externo implícito.
2. Para candidatos por redundância, agrupar/remover somente depois de confirmar que nenhuma asserção comportamental exclusiva se perde.
3. Priorizar melhoria de asserções fracas de ponta a ponta em vez de apagar integração que exercita várias camadas.
4. Validar qualquer alteração futura com o fluxo de `CONTRIBUTING.md` e os testes afetados.

## Síntese consolidada (triagem estática)

Até agora foram removidos 30 casos antigos e 3 verificações equivalentes foram consolidadas em uma; o saldo é **29 funções de teste a menos** (aprox. 1% da contagem inicial de 3.030 funções):

- Dois em `tests/models/test_base.py` (`test_initialize_called_correctly`, `test_model_call`) verificam overrides da classe fake, definidos dentro do teste, sem testar o comportamento de produto.
- Três em `tests/nn/modules/test_module.py` (`test_load_state_dict_empty`, `test_load_state_dict_with_extra_keys`, `test_load_state_dict_with_missing_keys`) usavam expressões tautológicas (`x is not None or x is None`) e foram removidos. Se esses cenários forem importantes, recriá-los com checagens de estado/semântica de chaves.
- Um em `tests/nn/modules/test_agent.py` (`test_inspect_model_execution_params`) tinha condição tautológica após confirmar que a saída era dict e foi removido. Se a inspeção de parâmetros exigir cobertura dedicada, criar um teste com expectativas concretas.

Além disso, os checks individuais duplicados do enum MCP, um loop de fluxo inteiramente implementado no teste, checks de herança redundantes e uma conversão nativa de string foram removidos ou consolidados. O teste de extensão de `ToolFlowControl` foi fortalecido; `test_toolflowcontrol_instantiation` foi mantido por verificar a instanciação da classe pública.

Os demais grupos sinalizados estão como **revisar**, incluindo asserções frouxas em multimodal/state_dict, verificações de constantes/enum, possíveis sobreposições entre providers e pares de testes de reasoning. Isso não é recomendação para remoção em bloco: em geral o fluxo exercitado tem valor, mas uma asserção específica pode não proteger o contrato declarado.

### Relatórios detalhados

- [Área core](core.md)
- [Models e providers](models.md)
- [Áreas especializadas](specialized.md)
- [Integração, DSL, utilitários e auth MCP](integration.md)
- [Fase 2: Agent/runtime/tools](phase2-agent-runtime.md)
- [Fase 2: workspace/persistência](phase2-workspace.md)
- [Fase 2: outros módulos](phase2-other.md)

## Outros sinais de baixo valor por assert

O relatório de integração também aponta checks redundantes ou fracos, sem classificar os casos completos como mortos: uma verificação negativa de `dotdict` implicada por exigir `str`; presença de `bfloat16` duplicada antes de acesso indexado; checks isolados de `callable` em decorators de retry; smoke do Mermaid que só exige não-`None`; e asserts live que só exigem texto não vazio. Em geral, a recomendação é fortalecer ou consolidar a asserção mantendo o cenário. Consulte [integration.md](integration.md) para a justificativa por área.

## Ação recomendada se for abrir uma etapa de limpeza

1. Se a cobertura dos cenários removidos de `load_state_dict` e `inspect_model_execution_params` for necessária, adicionar novos testes com expectativas concretas de estado e formato.
2. Fortalecer os casos “revisar” com asserts de saída observável e só depois consolidar casos equivalentes, preservando sync/async, provider e variações de entrada quando percorram caminhos diferentes.
