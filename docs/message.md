A classe msgflux.Message, inspirada no torch.Tensor, foi projetada para facilitar o tráfego de informações em grafos computacionais criados com módulos 'nn'. Um dos princípios centrais de sua concepção é permitir que cada Module tenha permissões específicas para ler e escrever em campos pré-definidos da Message. Isso proporciona uma estrutura organizada para o fluxo de dados entre diferentes componentes de um sistema.e quero experimentar todos, compor e comparar. Ace quero experimentar todos, compor e comparar. Ace quero experimentar todos, compor e comparar. Ac

A classe implementa os métodos set e get, que permitem criar e acessar dados na Message por meio de strings, oferecendo uma interface flexível e intuitiva. Além disso, campos padrão (default fields) são fornecidos para estruturar os dados de forma consistente.

# Estrutura e Campos Padrão
A inicialização da classe Message inclui os seguintes campos padrão, todos opcionais:
```python
def __init__(
    self,
    *,
    content: Optional[Union[str, OrderedDict[str, Any]]] = None,
    context: Optional[OrderedDict[str, Any]] = OrderedDict(),
    text: Optional[OrderedDict[str, Any]] = OrderedDict(),
    audios: Optional[OrderedDict[str, Any]] = OrderedDict(),
    images: Optional[OrderedDict[str, Any]] = OrderedDict(),
    videos: Optional[OrderedDict[str, Any]] = OrderedDict(),
    extra: Optional[OrderedDict[str, Any]] = OrderedDict(),
    user_id: Optional[str] = str(uuid4()),
    chat_id: Optional[str] = str(uuid4()),
):
```
content: Armazena o conteúdo principal da mensagem, podendo ser uma string ou um dicionário ordenado.

context: Campo sugerido para armazenar dados de contexto geradas por módulos.

text: Campo para textos adicionais estruturados como dicionário.

audios, images, videos: Campos para armazenar referências ou dados de mídia.

extra: Campo genérico para dados adicionais.

user_id e chat_id: Identificadores únicos gerados automaticamente (usando uuid4), caso não sejam fornecidos.

Message também autogera um 'execution_id'.

Para organizar o tráfego de informações, recomenda-se que os módulos escrevam suas saídas nos campos context, outputs e response. Já os campos de entrada (consumo de dados) podem ser livremente escolhidos conforme a necessidade do módulo.
Criando uma Message
Você pode criar uma instância de Message sem nenhum dado inicial ou com valores pré-definidos:
```python
import msgflux as mf

# Instância vazia
msg = mf.Message()

# Instância com dados iniciais
msg = mf.Message(content="Hello, world!", user_id="user123")
Inserindo Valores nos Campos Padrão
Os dados podem ser inseridos nos campos padrão usando o método set:
python
# Inserindo um conteúdo simples
msg.set("content", "What is Deep Learning?")

# Inserindo um arquivo de áudio
msg.set("audios.audio1", "audio.mp3")

# Inserindo uma URL de imagem
msg.set("images.google", "https://www.google.com/images/branding/googlelogo/1x/googlelogo_light_color_272x92dp.png")
Acesso Multinível (MultiLevel Set/Get)
A Message suporta uma estrutura hierárquica, permitindo o acesso a subcampos com notação de ponto (dot notation):
python
# Definindo um valor em um subcampo
msg.set("images.frontend.request", "https://huggingface.co/front/assets/huggingface_logo-noborder.svg")

# Acessando um nível superior
msg.get("images.frontend")
# Retorna: OrderedDict([('request', 'https://huggingface.co/front/assets/huggingface_logo-noborder.svg')])
```
Criando Novos Campos
Além dos campos padrão, é possível criar novos campos dinamicamente:
``` python
# Criando uma lista de PDFs
msg.set("pdfs", [])
msg.set("pdfs", "file.pdf")  # Substitui o valor anterior
msg.get("pdfs")
# Retorna: ['file.pdf']

# Criando um dicionário para CSVs
msg.set("csvs", {})
msg.set("csvs.backoffice", "3_quarter_stats.csv")
msg.get("csvs.backoffice")
# Retorna: "3_quarter_stats.csv"
Nota: O método set substitui o valor existente no campo especificado. Para adicionar itens a uma lista ou dicionário sem sobrescrever, é necessário manipular o conteúdo existente antes de usar set.
Rastreamento de Informações
Conforme os módulos produzem informações nos campos sugeridos (context, outputs, response), é possível rastrear o fluxo de dados com o método get_route():
python
msg.get_route()
# Retorna um registro das alterações feitas na Message (detalhes dependem da implementação)
Além disso, você pode verificar se um módulo específico já escreveu na Message:
python
msg.in_msg("module_name")
# Retorna True se "module_name" já modificou a Message, False caso contrário
```