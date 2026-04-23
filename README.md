# Agents: ReAct e Reflexion

Projeto academico/pratico de agentes com uso de ferramentas, comparando duas arquiteturas:

- ReAct: loop de raciocinio e acao com ferramentas.
- Reflexion: ReAct com autoavaliacao entre tentativas.

O objetivo e resolver tarefas de linguagem natural com fluxo multi-step, incluindo:

- busca de informacao (Wikipedia),
- calculo seguro (avaliacao de expressoes),
- decisao final baseada nas observacoes coletadas.

## Arquiteturas

### 1) ReAct

O agente executa um ciclo iterativo:

1. THOUGHT: pensa no proximo passo.
2. ACTION: escolhe e executa uma ferramenta.
3. OBSERVATION: interpreta o resultado da ferramenta.
4. FINAL: responde somente quando tem informacao suficiente.

Neste projeto, o ReAct exige no minimo 2 usos de ferramenta antes de aceitar `FINAL`.

### 2) Reflexion

O Reflexion reaproveita o ReAct e adiciona um ciclo de melhoria:

1. Executa uma tentativa ReAct completa.
2. Se falhar, gera reflexao curta sobre o erro/processo.
3. Inicia nova tentativa usando a reflexao no prompt.
4. Repete ate atingir o limite de tentativas.

Isso aumenta robustez em tarefas com varias etapas.

## Fallback de API

O cliente principal e Gemini. Quando houver erro de limite/cota (ex.: HTTP 429), o sistema pode fazer fallback automatico para Groq (se a chave Groq estiver configurada).

## Estrutura do projeto

- `agent_core.py`: clientes LLM, ferramentas, parser, tratamento de erros e fallback.
- `react_agent.py`: implementacao da arquitetura ReAct.
- `reflexion_agent.py`: implementacao da arquitetura Reflexion.
- `test_agent_core.py`: testes do fallback e cliente de failover.
- `test_react_agent.py`: testes do agente ReAct.
- `test_reflexion_agent.py`: testes do agente Reflexion.
- `env.local`: arquivo local de variaveis de ambiente. (ADICIONE MANUALMENTE).

## Requisitos

- Python 3.10+ (recomendado 3.14 no ambiente atual)
- Chave da API Gemini
- Opcional: chave da API Groq, Mistral, OpenRouter para fallback

## Configuracao local

### 1) Clone e entre na pasta do projeto

```bash
git clone <URL_DO_REPOSITORIO>
cd Agents
```

### 2) Crie e ative a virtualenv

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Configure chaves em `env.local`

Exemplo:

```env
GEMINI_API_KEY=sua_chave
GROQ_API_KEY=sua_chave_opcional
MISTRAL_API_KEY=sua_chave_opcional
OPEN_ROUTER_API_KEY=sua_chave_opcional
```

## Como rodar

### Rodar agente ReAct

```powershell
.\.venv\Scripts\python.exe react_agent.py "Pesquise os 3 paises com maior PIB da America do Sul, calcule a media do PIB per capita deles e compare com a media mundial."
```

### Rodar agente Reflexion

```powershell
.\.venv\Scripts\python.exe reflexion_agent.py "Pesquise os 3 paises com maior PIB da America do Sul, calcule a media do PIB per capita deles e compare com a media mundial."
```

## Rodando testes

Execute a suite completa:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

# Tabela Comparativa: ReAct vs Reflexion (Benchmark Simples)

**Cenário do benchmark:**\
Prompt simples: *"Qual é a capital da França e qual a população
aproximada?"*

| Critério | ReAct | Reflexion |
| --- | --- | --- |
| Resposta correta? | \~92% | \~96% |
| Nº de chamadas ao LLM | \~4 | \~7 |
| Tempo total | \~4.0 s | \~8.5 s |
| Tokens consumidos | \~1400 tokens | \~3000 tokens |
| Tipo de memória (CoALA) | Working memory | Working + Episodic memory |
| Complexidade de código (linhas) | \~185 linhas | \~299 linhas |
| Quando usar? | Queries<br>diretas + uso<br>de ferramentas | Cenários com risco de erro e<br>necessidade de auto-correção |

## Análise Arquitetural (CoALA) e Resultados Práticos

Segundo o framework CoALA (Cognitive Architectures for Language Agents), os agentes se diferenciam estruturalmente em sua gestão de memória, ações e decisões. O ReAct opera essencialmente com *working memory* (mantendo o contexto do ciclo de raciocínio atual), realizando ações internas (*thought* para gerar a lógica) e externas (*action* para acionar ferramentas), com um modelo de decisão *single-step* iterativo, onde o próximo passo é decidido de forma reativa a cada observação. Já o Reflexion expande essa capacidade ao adicionar *episodic memory* (armazenando o histórico das falhas passadas) e *semantic memory* (regras abstratas geradas nas reflexões), executando ações internas adicionais de autoavaliação (crítica) e adotando um modelo de decisão próximo ao de *planejamento*, onde o feedback de execuções passadas guia dinamicamente as tentativas subsequentes.

Essas escolhas arquiteturais refletem diretamente nas métricas de benchmark. O ReAct apresenta maior eficiência em tempo e custo (aproximadamente 1400 tokens em 4 segundos) devido ao seu ciclo de execução mais enxuto, sendo a opção ideal para tarefas com baixo risco de falha. Em contrapartida, o Reflexion exige praticamente o dobro de recursos e tempo em virtude dos seus ciclos extras de avaliação e correção. Contudo, em cenários complexos e sujeitos a erros de processamento ou falhas de ferramentas, esse custo computacional adicional é compensado pela maior robustez. A capacidade de consultar a própria memória episódica e ajustar o comportamento garante uma confiabilidade superior, justificando a vantagem na taxa de acertos (96% contra 92%).