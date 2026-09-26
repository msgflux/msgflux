# Auditoria de testes: Agent/runtime/workspace/tools (área core)

Escopo desta parte: testes diretamente sob `tests/`, `tests/tools/`, `tests/tools/builtin/`, `tests/auto/`, `tests/dsl/`, `tests/utils/` e `tests/integration/`; excluídos `models`, `data`, `nn`, `protocols`, `telemetry` e `generation`, conforme a divisão da tarefa. A revisão buscou casos sem asserts, asserts puramente tautológicos, duplicação por parametrização ausente e testes de comportamento sem ligação com um contrato observável. A busca sintática por funções sem `assert` foi usada apenas como triagem: muitos desses casos usam `pytest.raises` e verificam corretamente rejeições.

## Candidato forte

Nenhum identificado nesta parte com evidência suficiente para recomendar remoção. Os casos que parecem simples em leitura rápida, em geral, fazem verificações de borda, segurança, erro, persistência ou formato que correspondem a caminhos reais da implementação. Ausência de `assert` textual, mock ou teste isolado não foi considerada evidência de inutilidade.

## Revisar

- `tests/test_agent_image_detail.py::TestAgentImageDetail.test_agent_init_with_image_detail_high` e `::test_agent_init_with_image_detail_low` — **Revisar (redundância de manutenção, não teste morto).** Cada teste só verifica o valor de `config["image_block_kwargs"]["detail"]` imediatamente após a construção. A implementação em `src/msgflux/nn/modules/agent/configuration.py::_set_config` copia a configuração; o caminho de uso em `src/msgflux/nn/modules/agent/inputs.py::_format_image_input` encaminha os kwargs a `Image`. Os dois valores exercitam entradas distintas e ambos têm consumidores; ainda assim, podem ser parametrizados ou incorporados a um teste de ponta a ponta existente para reduzir boilerplate. Risco de remoção: perde cobertura explícita de que ambos os valores são aceitos e preservados pela configuração. Recomendação: manter a cobertura dos dois valores, consolidando apenas se a suíte já cobrir o encaminhamento final.

## Úteis (não recomendar remoção)

- `tests/test_agent_image_detail.py::TestAgentImageDetail.test_agent_init_with_invalid_image_detail` — verifica deliberadamente que Agent aceita e preserva o valor e deixa validação para a camada de bloco. O comentário corresponde à divisão de responsabilidade; não é teste sem propósito só por aceitar entrada inválida.
- `tests/test_no_legacy_httpx.py::test_source_does_not_import_legacy_httpx` e `::test_project_does_not_declare_legacy_httpx_dependency` — guardas arquiteturais contra regressão em imports e dependências; são varreduras estáticas, mas falham diante de regressões concretas.
- `tests/test_tool_config_decorator.py::test_tool_config_with_agent_as_tool` — confirma propagação da configuração até o registro da ferramenta, além do atributo local. É um consumidor real que uma verificação de atributo isolada não cobriria.
- `tests/test_apply_patch.py::test_v4a_parser` (parametrizado) — cobre resultados distintos para criação, substituição, CRLF, ausência de newline e contexto/EOF; não são repetições equivalentes.
- `tests/test_workspace_query.py::test_query_budget_exhaustion_fails_closed` e `::test_ignored_directory_is_pruned_without_grants_to_its_contents` — cobrem limites e autorização em buscas no workspace; os nomes curtos escondem invariantes de segurança, então não classificar como fúteis sem analisar os asserts.

## Observações da triagem

- A AST encontrou muitos `test_*` sem `assert` explícito, mas os casos inspecionados usam `pytest.raises(...)` para verificar exceções esperadas (por exemplo, validações de configuração, IDs, caminhos e limites). Não recomendo remoção baseada nessa heurística.
- Há testes de configuração que se limitam a ler o atributo configurado, mas isso pode ser precisamente o contrato público do decorator/config. Tratar como teste de baixo valor requer verificar também se o comportamento downstream está coberto; não há base nesta revisão para marcar um arquivo inteiro como morto.
- Esta conclusão é conservadora e localizada ao escopo desta parte; não equivale a uma auditoria de mutação ou cobertura por linha.
