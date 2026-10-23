# Auditoria fase 2: Agent, runtimes, hooks, tools e execuções

## Escopo e método

Revisão estática da suíte de runtime selecionada em `tests/`: agentes e execução de geração, tarefas, hooks, ferramentas e execução de processos/workspace. Foram comparados os casos com as implementações em `src/msgflux/nn/modules/agent/`, `src/msgflux/generation/`, `src/msgflux/tasks/`, `src/msgflux/runtime/` e `src/msgflux/tools/`, além dos relatórios anteriores. Os seis casos já removidos e qualquer achado já descrito em `core.md` foram excluídos desta lista.

O inventário geral anterior contava aproximadamente 3.030 funções de teste. Nesta área, a triagem de nomes e leitura contextual abrangeu cerca de 500 funções; a leitura aprofundada de implementação e asserts concentrou-se em aproximadamente 180 casos em ~45 arquivos, incluindo os casos limítrofes citados abaixo. Contagens são aproximadas e textuais, não uma medição do pytest. Não houve execução de testes, remoção temporária, cobertura ou mutation testing.

## Removidos após solicitação do usuário

- `tests/tools/test_toolflowcontrol.py`: `test_toolflowcontrol_inheritance`, `test_toolflowcontrol_multiple_inheritance`, `test_toolflowcontrol_method_addition`, `test_toolflowcontrol_state_management`, `test_toolflowcontrol_callable_subclass`, `test_toolflowcontrol_with_properties`, `test_toolflowcontrol_with_class_variables` e `test_toolflowcontrol_isolation` (8 casos). Cada um definia uma subclasse local e testava recursos gerais da linguagem Python (herança, métodos, estado que a própria classe criou, `__call__`, property, variáveis de classe e isolamento de atributos de instância). Nenhum chamava o contrato de `ToolFlowControl` (`extract_flow_result`, `inject_results`, `build_history` ou variantes async), nem o integrava ao runtime. O contrato do produto é exercitado em `tests/generation/test_control_flow.py`, `tests/generation/test_reasoning.py` e testes de integração de provider/Agent. Os oito casos foram removidos.

- `tests/tools/test_toolflowcontrol.py::test_toolflowcontrol_has_docstring` (1 caso) verificava texto literal de docstring, incluindo frase que não é parte do contrato executável. Uma edição editorial incidental quebraria o teste sem indicar regressão. Removido.

O mesmo arquivo mantém `test_toolflowcontrol_instantiation`, que instancia a base e verifica `isinstance`. É um candidato marginalmente redundante (não há estado ou inicialização na base), mas preserva a intenção de que a classe pública seja instanciável. Foram removidos **9 casos** deste arquivo; esse teste foi mantido.

## Asserções fracas, preservar o cenário

- `tests/test_agent_generation_schema.py::test_agent_with_generation_schema_creates_output_with_merged_annotations`: `assert final_answer_type is not None` é fraco e não valida o tipo prometido pelo comentário. O teste, contudo, também verifica herança e merge de anotações usados por `Agent`; remover o teste inteiro perderia cobertura útil. Substituir o check por expectativa concreta do tipo produzido pela assinatura.
- `tests/test_agent_generation_schema.py::test_agent_with_generation_schema_optional_final_answer`: o teste valida que `final_answer` difere de `str`, mas não define qual tipo esperado. Manter o caso por exercitar a substituição de campo herdado; fortalecer a asserção.
- `tests/test_agent_named_kwargs.py` contém checks `result is not None`/`params is not None` em alguns caminhos. São apenas checks de existência, mas os casos percorrem resolução de parâmetros nomeados em entradas/configurações diferentes. Não há evidência de equivalência suficiente para apagar os casos; elevar asserções aos valores e chamadas esperados.

## Áreas examinadas sem recomendação de remoção

- `tests/test_async_hooks.py` não duplica `tests/utils/test_hooks.py`: o primeiro cobre composição/ordem e efeitos de hooks síncronos e assíncronos na execução de `Module`; o segundo cobre ciclo de vida de `RemovableHandle`. O teste `test_no_hooks_fast_path` também exercita a chamada async sem hook.
- Os testes de Agent que aparentam verificar apenas existência de `state`, `result` ou `None` frequentemente protegem estados de checkpoint, retomada, recusa, streaming e ausência de duplicação de chamadas; não são candidatos por inspeção do assert isolado.
- `tests/test_executor.py`, `tests/test_execution_environment.py`, `tests/test_runtime_permissions.py`, `tests/test_tool_permissions.py`, `tests/test_task_runtime.py` e os testes de adapters exercitam lifecycle, isolamento concorrente, cancelamento, autorização e não repetição de efeitos. São contratos de regressão relevantes e não devem ser reduzidos por meta percentual.
- `tests/tools/test_toolflowcontrol.py::test_toolflowcontrol_state_management` testa um dicionário criado dentro do teste, e `::test_toolflowcontrol_isolation` testa contador local; ambos não cobrem estado interno do pacote.

## Recomendação sobre a meta de 10%

Uma meta de remover 10% da suíte significaria cerca de 303 funções segundo a contagem anterior. Esta rodada encontrou nove casos com justificativa suficientemente concreta, todos concentrados num teste didático de propriedades genéricas de Python. Isso não sustenta remover centenas de casos com segurança: muitos testes curtos protegem fronteiras de segurança, persistência, concorrência, erros ou compatibilidade. Tratar 10% como quota de remoção pressionaria a apagar contratos reais. Recomendo medir a redução apenas como resultado de uma auditoria comportamental (idealmente com mutation testing/cobertura de mutação por módulos), não como critério de aceite. Os candidatos fortes devem ser removidos em grupos pequenos e validados na implementação da tarefa de limpeza.
