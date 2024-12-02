## Whisper
Whisper é um modelo de reconhecimento de fala de uso geral. Ele foi treinado com um grande conjunto de dados de áudio diversificado e é um modelo multitarefa capaz de realizar reconhecimento de fala multilíngue, tradução de fala e identificação de idioma.

## Método

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

O modelo utiliza um Transformer de sequência para sequência treinado em várias tarefas de processamento de fala, incluindo reconhecimento de fala multilíngue, tradução de fala, identificação de idioma falado e detecção de atividade de voz. Essas tarefas são representadas como uma sequência de tokens prevista pelo decodificador, permitindo que um único modelo substitua várias etapas de um pipeline tradicional de processamento de fala. O formato de treinamento multitarefa usa um conjunto de tokens especiais que funcionam como especificadores de tarefa ou alvos de classificação.

## Como utilizar!

O treinamento e os testes foram realizados usando Python 3.9.9 e PyTorch 1.10.1. No entanto, o código é compatível com Python 3.8–3.11 e versões recentes do PyTorch. O projeto também depende de alguns pacotes Python, como tiktoken da OpenAI para tokenização rápida. Para instalar a última versão do Whisper:
        
        pip install -U openai-whisper

Para instalar diretamente do repositório GitHub com todas as depedências:

        pip install git+https://github.com/openai/whisper.git

Para atualizar o pacote execute:

        pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

É necessário o utilitário de linha de comando [`ffmpeg`](https://ffmpeg.org/), que pode ser instalado pelos gerenciadores de pacotes:

```bash

# Ubuntu ou Debian
sudo apt update && sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# Windows (Scoop)
scoop install ffmpeg

```

Também pode ser necessário instalar [`rust`](http://rust-lang.org). Caso o [tiktoken](https://github.com/openai/tiktoken)não fornecça um pre-built, siga [Getting started page](https://www.rust-lang.org/learn/get-started) de introdução ao Rust para configurar o ambiente. Adicionalmente, configure a variável de ambiente `PATH`
`export PATH="$HOME/.cargo/bin:$PATH"`. Se ocorrer o erro `No module named 'setuptools_rust'` `setuptools_rust'`, instale o módulo com:

```bash
pip install setuptools-rust
```

## Modelos disponíveis e idiomas

O Whisper oferece seis tamanhos de modelo, com versões somente em inglês ou multilíngues, balanceando velocidade e precisão.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |


Os modelos `tiny.en` e `base.en`, voltados apenas para inglês, apresentam melhor desempenho. Para tarefas multilíngues, o modelo turbo oferece maior velocidade com uma leve redução na precisão. Abaixo, uma comparação de taxas de erro (WER e CER) por idioma para os modelos `large-v3` e `large-v2`:

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

## Uso via linha de comando

O comando a seguir irá transcrever a fala em arquivos de áudio, utilizando o modelo `turbo`:

     whisper audio.flac audio.mp3 audio.wav --model turbo

A configuração padrão (que seleciona o modelo `turbo`) funciona bem para transcrição em inglês. Para transcrever um arquivo de áudio contendo fala em um idioma diferente do inglês, você pode especificar o idioma usando a opção `--language`:

     whisper japanese.wav --language Japanese

Adicionando `--task translate`, a fala será traduzida para o inglês:

    whisper japanese.wav --language Japanese --task translate

Execute o seguinte comando para visualizar todas as opções disponíveis:

    whisper --help

Consulte [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) para a lista de todos os idiomas disponíveis.

## Uso com Python

A transcrição também pode ser realizada diretamente no Python:

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

Internamente, o método `transcribe()` lê o arquivo inteiro e processa o áudio em janelas deslizantes de 30 segundos, realizando previsões sequenciais autoregressivas em cada janela.

Abaixo está um exemplo de uso de `whisper.detect_language()` e `whisper.decode()`, que fornecem acesso de nível mais baixo ao modelo:

```python
import whisper

model = whisper.load_model("turbo")

# Carregar o áudio e ajustá-lo para caber em 30 segundos
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# Criar um espectrograma log-Mel e movê-lo para o mesmo dispositivo do modelo
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# Detectar o idioma falado
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Decodificar o áudio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# Imprimir o texto reconhecido
print(result.text)
```
## Transformer
![Imagem do WhatsApp de 2024-12-02 à(s) 17 26 47_d4b0dc98](https://github.com/user-attachments/assets/7c55cc3d-5cb9-466b-b7d4-5fa45d948972)

## Seq2Seq
![Imagem do WhatsApp de 2024-12-02 à(s) 17 26 22_5b56a549](https://github.com/user-attachments/assets/d7aa6c22-eef8-4f58-abf0-7bfe303a71c9)
## Mais exemplos


Por favor, utilize a [🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) em Discussões para compartilhar mais exemplos de uso do Whisper e extensões de terceiros, como demonstrações web, integrações com outras ferramentas, ports para diferentes plataformas, etc.
