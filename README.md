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
- Opcional: chave da API Groq para fallback

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
GEMINI_API_KEY=sua_chave_gemini
GROQ_API_KEY=sua_chave_groq_opcional
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
| Tipo de memória (CoALA) | Working memory | Episódica + semântica |
| Complexidade de código (linhas) | \~185 linhas | \~299 linhas |
| Quando usar? | Queries<br>diretas + uso<br>de ferramentas | Cenários com risco de erro e<br>necessidade de auto-correção |

## Classificação CoALA

### ReAct

-   Memória:
    -   Working memory (estado no prompt, sem persistência)
-   Ações:
    -   Internas: cadeia de raciocínio (thought)
    -   Externas: chamadas a ferramentas (action)
-   Tipo de decisão:
    -   Multi-step reativo (sem planejamento explícito)

### Reflexion

-   Memória:
    -   Episódica: armazena tentativas anteriores
    -   Semântica: abstrai erros recorrentes
-   Ações:
    -   Internas: reflexão/autoavaliação
    -   Externas: chamadas ao ambiente (opcional)
-   Tipo de decisão:
    -   Multi-step com loop de feedback (tentativa → crítica → ajuste)

## Interpretação dos números

-   O **ReAct** resolve rapidamente com poucas chamadas, pois executa
    apenas um ciclo de raciocínio + ação.
-   O **Reflexion** repete ciclos internos (geração → crítica →
    refinamento), aumentando custo e tempo.
-   O ganho de acurácia no Reflexion vem da **redução de erros por
    autoavaliação**, mesmo em prompts simples.

## Conclusão prática

-   Use **ReAct** quando:
    -   Latência e custo são críticos
    -   A tarefa é bem definida
-   Use **Reflexion** quando:
    -   Há ambiguidade ou risco de erro
    -   Qualidade da resposta é mais importante que custo