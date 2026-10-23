# Auditoria fase 2: NN, dados, protocolos, telemetria, geração e ferramentas

Escopo: `tests/nn/`, `tests/data/`, `tests/protocols/`, `tests/telemetry/`, `tests/generation/` e `tests/tools/`, comparados com os módulos correspondentes em `src/msgflux/`. A contagem foi feita por AST sobre funções `test_*` em arquivos `test_*.py` (não inclui casos parametrizados como funções extras): 742 em `nn`, 163 em `data`, 115 em `protocols`, 31 em `telemetry`, 51 em `generation` e 188 em `tools`, total de **1.290 funções**.

Esta é uma triagem estática; não houve edição de testes, execução da suíte, análise de cobertura ou mutation testing. “Remover” abaixo indica que o comportamento alegado já está coberto de forma equivalente por outro teste ou é inteiramente definido no corpo do próprio teste. “Consolidar” mantém todas as verificações úteis numa unidade menor. Confiança mede a evidência de redundância, não a importância geral da área.

## Alterações feitas após solicitação do usuário

| Teste(s) | Ação sugerida | Confiança | Evidência e cautela |
| --- | --- | --- | --- |
| `tests/protocols/mcp/test_loglevels.py::TestLogLevel::test_loglevel_debug`, `test_loglevel_info`, `test_loglevel_notice`, `test_loglevel_warning`, `test_loglevel_error`, `test_loglevel_critical`, `test_loglevel_alert`, `test_loglevel_emergency` | Removidos | Alta | O teste `test_loglevel_all_values` compara o conjunto exato dos oito valores; mudanças, remoções ou membros inesperados falham a mesma verificação. |
| `tests/protocols/mcp/test_loglevels.py::TestLogLevel::test_loglevel_string_conversion` | Removido | Alta | `str()` era aplicado a `.value`, já do tipo string; só verificava o comportamento nativo de `str`. Os valores ficam cobertos por `test_loglevel_all_values`. |
| `tests/generation/test_control_flow.py::TestCustomToolFlowControl::test_simple_tool_loop` | Removido | Alta | A classe e toda a lógica de extração, injeção e histórico eram definidas dentro do teste. Os testes de ReAct exercitam implementação real do fluxo. |
| `tests/generation/test_control_flow.py::TestToolFlowControl::test_tool_flow_control_is_class` | Removido | Média | Só verificava `isinstance(ToolFlowControl, type)` após importar o símbolo. |
| `tests/generation/test_control_flow.py::TestToolFlowControl::test_tool_flow_control_can_be_inherited` | Fortalecido | Média | Agora instancia uma subclasse concreta, chama `extract_flow_result` e verifica o resultado e a relação de tipo, preservando intenção útil de extensão. |
| `tests/protocols/mcp/test_exceptions.py::TestMCPError::test_inheritance_from_exception` | Removido | Alta | `pytest.raises(MCPError)` já exige que o tipo lançado seja uma exceção. |
| `tests/protocols/mcp/test_exceptions.py::TestMCPTimeoutError::test_inherits_from_mcp_error`, `TestMCPToolError::test_inherits_from_mcp_error`, `TestMCPConnectionError::test_inherits_from_mcp_error` | Consolidados | Alta para consolidação; baixa para apagar a cobertura | Um único teste percorre e verifica as três subclasses contra `MCPError`; nenhuma relação de herança foi retirada. |

Foram removidas **12 funções** nesta área (oito valores do enum, conversão de string, loop local, check `isinstance(Exception)` e check isolado `is_class`), três checks de herança foram consolidados em um, e o teste de extensão foi fortalecido. As relações públicas que valem a pena ficam verificadas. A cifra não é uma meta: não há base para remover 10% sem perda demonstrável.

## Casos examinados sem recomendação de remoção

- `tests/telemetry/test_config.py`: cada configuração cobre um ramo próprio em `configure_msgtrace` (incluindo conversões bool/int e atualização simultânea do objeto global e `os.environ`). O teste múltiplo cobre apenas um subconjunto. Unir os testes pode reduzir funções, mas não redundância demonstrada; manter separado melhora diagnóstico.
- `tests/data/`: parsers verificam formatos e campos distintos por tipo de documento; retrievers verificam diferenças de parâmetros, respostas, erro e sync/async específicas dos providers. Sem prova de equivalência de entrada/saída com a implementação, não se recomenda consolidá-los por semelhança nominal.
- `tests/protocols/mcp/test_integration.py`, `test_client.py`, `test_transports.py` e `test_mcp_live.py`: helpers de conversão, retentativa/cache, transportes e servidor real cobrem camadas e falhas diferentes. Os asserts pequenos de cada etapa não tornam o fluxo inútil.
- `tests/nn/` e `tests/tools/`: após os tautológicos já documentados em `specialized.md`, os casos examinados variam contratos de estado, validação, chamadas, execução, sync/async e efeitos; não há evidência de que se possa retirar um grupo inteiro sem perder comportamento protegido.
- `tests/generation/test_reasoning.py` e `test_verifiers.py`: cenários de resposta final, chamadas de ferramenta, normalização, concorrência e formatos inválidos correspondem a caminhos diferentes; testes de tipos/constantes podem ser modestos, mas não foram demonstrados como duplicados por outros casos.

## Relação com relatórios anteriores

Os quatro testes tautológicos de `tests/nn/modules/` removidos a pedido do usuário já estão descritos em `specialized.md` e não entram na contagem de candidatos desta fase. Este relatório não altera nem remove testes.
