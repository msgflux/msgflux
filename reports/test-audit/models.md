# Auditoria de testes: `tests/models/`

## Escopo e critério

Revisei os testes de `tests/models/` e `tests/models/providers/` comparando o que eles afirmam com as implementações em `src/msgflux/models/` e `src/msgflux/models/providers/`. Classifiquei como candidato forte apenas teste cujo resultado depende essencialmente de um stub definido no próprio teste, sem exercitar comportamento do produto. Testes com mocks continuam úteis quando verificam contratos de request/response, serialização, credenciais, erros, streaming ou seleção/registro de provider.

## Removidos após solicitação do usuário

| Teste | Classificação | Evidência | Risco e recomendação |
| --- | --- | --- | --- |
| `tests/models/test_base.py::TestBaseModel::test_initialize_called_correctly` | Removido | `ConcreteModel._initialize()` era implementado no próprio teste como `self.client = "initialized_client"`; o caso chamava esse stub e verificava o efeito definido pelo próprio double, sem exercitar `BaseModel`. | Baixo risco para cobertura do produto. |
| `tests/models/test_base.py::TestBaseModel::test_model_call` | Removido | `ConcreteModel.__call__` era definido no próprio teste retornando `{"result": "success"}`; a asserção repetia esse valor e a classe base não fornece comportamento `__call__`. | Baixo risco para cobertura do produto. |

## Revisar, não remover automaticamente

| Teste(s) | Classificação | Evidência | Risco e recomendação |
| --- | --- | --- | --- |
| `tests/models/test_base.py::TestBaseModel::test_base_model_msgflux_type`, `test_base_model_to_ignore`, `test_instance_type`, `test_get_model_info` | Revisar | As asserções são curtas, mas observam metadados e a lista de campos excluídos de serialização herdados por todos os modelos. `BaseModel.to_ignore` contém campos sensíveis e recursos runtime (`_api_key`, clientes e transports); mudanças acidentais podem vazar estado ou quebrar reconstrução. | Baixo a médio risco de manter; são contratos internos importantes. Preferir consolidar em testes parametrizados somente se isso não reduzir a legibilidade dos campos protegidos. |
| `tests/models/providers/test_brave.py::test_config`, `tests/models/providers/test_exa.py::test_config`, `test_config_with_research_model` | Revisar | São checks superficiais de configuração, mas os provedores declaram valores diferentes em `ProviderEnvBase` (`api_key_env`, `base_url`, `base_url_env`) e Exa documenta IDs de modelo especiais. | Baixo risco se mantidos: cobrem defaults que uma regressão de declaração quebraria. O teste `test_config_with_research_model` é o mais fraco, pois apenas verifica que `model_id` preserva o argumento passado; removê-lo seria razoável se a classe não adicionar normalização/roteamento por ID. |
| Testes de inicialização, URL, API key e registro em `test_fireworks.py`, `test_nvidia.py`, `test_baseten.py` (ex.: `test_*_defaults_to_chat_completions`, `test_*_models_registered`, `test_*_resolves_through_model_factory`) | Revisar por possível sobreposição | Há repetição entre provedores OpenAI-compatible e alguns testes checam o mesmo contrato em níveis próximos; contudo cada provider tem configuração/registro independente e as resoluções pela factory cobrem integração distinta da instanciação direta. | Médio risco de remoção em lote: poderia deixar provider sem cobertura de registro/configuração. Antes de consolidar, verificar se existe teste parametrizado equivalente sobre todas as classes e conservar pelo menos um teste de factory/registro por provider. |

## Testes que aparentam úteis

- `tests/models/test_openai_sdk_independence.py::test_openai_models_import_and_initialize_without_openai_sdk`: valida independência de dependência opcional em processo isolado.
- `tests/models/test_multipart.py::*`: verifica nomes, MIME types, conversão de bytes e codificação multipart consumida por endpoints de áudio.
- `tests/models/test_session.py::*` e `tests/models/test_timing.py::*`: validam headers dependentes de contexto e métricas first-output/latência, inclusive idempotência.
- Testes de providers que verificam payload, streaming, credenciais resolvidas por request, erros e serialização de transports (por exemplo `test_openai_embeddings.py`, `test_openai_image_edit.py`, `test_openai_moderation.py`, `test_openai_speech.py`, `test_ollama.py`) exercitam contratos externos e regressões relevantes apesar de usarem mocks.

## Limite desta triagem

Este relatório aponta candidatos, não propõe exclusão automática. Os únicos casos fortes localizados nesta passada são dois testes em `test_base.py` que testam overrides da própria classe fake. Os testes de configuração curtos de provider parecem frágeis/superficiais, mas não são claramente inúteis porque protegem configuração independente por provider. Não há recomendação para remover testes de serialização, erros, rede simulada ou contratos de provider só por serem unitários.
