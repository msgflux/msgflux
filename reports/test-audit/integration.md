# Integração, DSL, utilitários e autenticação MCP

## Escopo

Leitura estática caso a caso dos testes nestas quatro áreas, confrontada com as funções e classes exercitadas. Não executei testes, cobertura nem mutation testing. “Fraco” indica assert estreito ou redundante; não basta para recomendar apagar o caso completo.

## Integração

| Casos | Classificação | Evidência / limite |
|---|---|---|
| tests/integration/test_reasoning.py: testes sync/async de geração, has_reasoning e sem reasoning | Útil | Cobrem respostas do provider Groq e presença/ausência do reasoning. Os casos de has_reasoning e conteúdo têm sobreposição parcial, mas afirmam atributos distintos do objeto. Dependem de credenciais e resposta real. |
| tests/integration/test_reasoning.py: streaming sync/async, ordem de eventos e consume_reasoning primeiro | Útil | Conferem metadados, eventos, filas e ordens de consumo diferentes. Asserções de texto não vazio são fracas e variáveis, mas não tornam o fluxo de streaming redundante. |
| test_agent_reasoning.py: test_agent_sync_text_with_reasoning, test_agent_sync_no_reasoning_in_response, test_agent_async_text_with_reasoning | Parcialmente redundante | O teste sync que afirma str já torna falsa a condição de dotdict. Ambos usam o mesmo tipo de fixture/pergunta. A asserção negativa de dotdict não acrescenta distinção observável; porém sync e async percorrem dispatches distintos, e o contrato sem wrapper continua útil. |
| test_agent_reasoning.py: reasoning_in_response sync/async | Útil, assert fraco | Garantem dotdict com answer/reasoning. len > 0 não confirma semântica estável do conteúdo. |
| test_agent_reasoning.py: streaming sync/async e eventos | Útil | Verificam tipo, sinalização, campos finais e consumo antecipado/separado de reasoning. Compartilhar o início da chamada não torna os asserts de evento e conclusão equivalentes. |
| test_agent_reasoning.py: tool call com reasoning sync/async e saída em dotdict | Útil | Integram chamada de ferramenta com reasoning; conteúdo do modelo varia, mas execução e formato são observáveis. |
| test_guard_moderation.py: input seguro/perigoso sync/async | Útil | Exercitam Guard com modelo de moderação real, incluindo mensagem bloqueada. Quatro combinações cobrem modos e resultados; dependem do provider. |
| test_tool_dict_e2e.py (4 casos) | Útil | Agent chama ferramenta com dict, com efeito/argumentos observáveis, sync/async e variantes ReAct. |
| test_openai_tool_runtime_e2e.py, test_openai_compaction_e2e.py, test_tool_msgspec_struct_e2e.py, test_tts_stream_next_chunk.py | Útil, opt-in | Cobrem loop de ferramenta, handle capturado, compactação/continuação, parâmetro Struct e consumo de TTS. Requerem credenciais/job opt-in; skip padrão não é evidência de teste morto. |
| test_tool_offload_e2e.py::test_live_provider_shell_output_offload_and_followup | Útil, opt-in | Verifica offload de saída grande e uso posterior. Valor prático requer ambiente que o execute. |
| test_live_agent_provider_matrix.py::test_live_agent_tool_stream_checkpoint_and_thread_replay e test_live_agent_stress_large_offload_bounded_events_sqlite | Útil, opt-in/stress | Cobrem provider, eventos, ferramenta, checkpoint/replay, payload grande e SQLite. O modo stress é propositalmente caro; verificar job live periódico. |
| test_live_agent_provider_matrix.py: cinco testes model factory/configuração | Útil, determinístico | Monkeypatch cobre seleção de provider registrado, fallback OpenAI compatível e erros/configurações incompletas, sem chamada externa. |
| test_live_background_inbox_resolution.py::test_live_resumed_agenttool_routes_message_without_inbox_map | Útil, opt-in | Verifica roteamento para subagente ativo após retomada, cenário de concorrência live. Requer job com credenciais. |

## DSL

| Casos | Classificação | Evidência / limite |
|---|---|---|
| tests/dsl/test_inline.py (todos) | Útil; sem candidato forte | Cobrem fluxos sync/async, sequência, paralelismo, condições e operadores, loops/limite de iterações, caminhos pontuados, módulos múltiplos, falhas de tarefa/parser e multibranch. Casos simples com nomes próximos exercitam configurações/resultados diferentes; não encontrei equivalência demonstrada. Booleanos, floats e casos zero/negativo são limites, não duplicatas óbvias. |
| tests/dsl/test_signature.py (todos) | Útil; sem candidato forte | Verificam FieldInfo, ordem, anotações, templates, descrições, campos vazios, tipos complexos e mídia. Strings completas são sensíveis a mudanças de apresentação, mas são saída de prompt consumida pelo sistema. Exemplos similares não afirmam a mesma entrada/saída. |

## Utilitários

| Arquivo / casos | Classificação | Evidência / limite |
|---|---|---|
| test_logging.py (2) | Útil | Formatter para linha simples e multilinha. |
| test_hint_to_schema_dict.py (13) | Útil | Tipos aninhados, Any, tipos inválidos, obrigatoriedade e estrutura do schema. |
| test_validation.py (3), test_convert.py (3) | Útil | Resultados positivos/negativos para validação e três transformações distintas. |
| test_imports.py::test_import_module_from_lib | Útil, assert positivo fraco | Importa símbolo real e exige ImportError/AttributeError nos dois erros. O assert de sucesso é só truthiness, mas os erros protegem comportamento real. |
| test_msgspec.py | Útil | Round-trip, JSON inválido, Struct por schema/assinatura, persistência, optionalidade, conversões/restauração OpenAI, wrappers inválidos, Union e chaves Enum. Não vi casos equivalentes. |
| test_chat.py | Útil | Mensagens/mídias, add/clear, schema de ferramenta, msgspec, docstring e adaptação de áudio. As variantes de mídia são formas diferentes de entrada. |
| test_encode.py | Útil | URL, arquivo, data/base64, bytes, IO, fallback de filename e caminho async mockado; origens e resultados distintos. |
| test_inspect.py, test_templates.py, test_hooks.py, test_common.py, test_console.py | Útil | Exercitam inspeção, templates/tipos, lifecycle de handle, sintaxe/placeholders, saída e cores. A simplicidade não torna esses contratos fúteis. |
| test_pooling.py (4) | Útil | Média, máximo, CLS e estratégia inválida cobrem ramos diferentes. |
| test_torch.py::test_torch_dtype_map_contains_expected_types e test_torch_dtype_map_bfloat16 | Assert parcialmente redundante | A presença bfloat16 no primeiro é implicada pelo acesso indexado no segundo. Presença de float16/float32 e tipo dos valores são verificações separadas; não excluir o arquivo. importorskip condiciona cobertura à instalação do Torch. |
| test_tenacity.py::test_default_tool_retry_exists e test_default_model_retry_exists | Fraco | callable acrescenta pouca evidência, pois os decorators são importados e usados nos testes seguintes. Remoção depende de serem ou não API pública. |
| test_tenacity.py: sucesso dos defaults e casos de apply_retry | Útil, com lacuna | Contadores distinguem default, desativação, customização e ausência de retry. Os dois testes de decorator default são quase paralelos e não provocam erro transitório: demonstram só sucesso sem retry. Oportunidade de parametrizar/fortalecer, não prova de teste morto. |
| test_xml.py (2) | Útil, estreito | Retornos exatos cobrem tag de saída padrão e explícita. |
| test_mermaid.py::test_plot_mermaid_with_simple_diagram | Fraco, não provado morto | Só afirma resultado não None e faz skip sem dependência. É smoke test real quando instalado, porém não valida imagem/conteúdo; melhorar a asserção é mais justificado que excluir por evidência atual. |

## Autenticação MCP

| Casos | Classificação | Evidência / limite |
|---|---|---|
| BearerTokenAuth (aplicar, token type, update, refresh) | Útil | Header/esquema, mutação e callback async com expiração e atualização efetiva. |
| APIKeyAuth (header default/custom, prefixo, update) | Útil | Opções de header e prefixo produzem valores diferentes. |
| BasicAuth (aplicar, atualizar) | Útil, alguma sobreposição | Ambos decodificam credenciais; o segundo também cobre mutação. Não são equivalentes. |
| OAuth2Auth (apply, refresh, update tokens) | Útil | Header, callback/refresh token, expiração e update. |
| CustomHeaderAuth (static, callback, precedência, update) | Útil | Testa fontes dinâmicas/estáticas, precedência e substituição observável. |
| BaseAuthExpiration (set, expired, auth info) | Útil | Estados e metadados distintos. |

## Candidatos concretos e limites

Não encontrei teste completo que possa ser chamado de morto com evidência estática. Candidatos de assert/granularidade: (1) a negativa de dotdict é implicada pelo teste sync que exige str; (2) presença de bfloat16 repetida antes do acesso indexado; (3) callable isolado dos decorators de retry; (4) smoke test Mermaid que só exige não-None; (5) asserts de resposta live que só exigem texto não vazio. Os itens 1–3 são redundâncias de asserção, e 4–5 são oportunidades para melhorar expectativas. Nenhum justifica remover uma área inteira.

Não executei testes, cobertura, mutation testing ou conferência dos jobs que rodam integrações live; por isso, não afirmo cobertura operacional dessas suites. Providers podem produzir conteúdo variável e chamadas podem ser pagas. Para uma decisão de exclusão mais forte, conferir jobs live e mutações/cobertura de branches. Nenhum teste foi removido ou alterado nesta auditoria.
