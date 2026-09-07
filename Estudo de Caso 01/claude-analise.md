\# Relatório de Revisão Consolidado — Estudo de Caso 01 (IMC PPGEE 2016-2 vs 2017-2)



Documento revisado: `Estudo\_de\_caso\_01.Rmd`

Escopo: revisão técnica, conceitual e de conformidade. Sem autorização de edição do manuscrito.



Este relatório funde duas revisões independentes do mesmo arquivo: uma passada técnica/adversarial (verificada por execução do código R contra os dados reais) e uma passada de conformidade acadêmica em 8 etapas. Cada achado indica sua origem:



\- \*\*\[AMBOS]\*\* — os dois relatórios levantaram, de forma independente (maior confiança)

\- \*\*\[TÉCNICO]\*\* — só a passada técnica/adversarial

\- \*\*\[CONFORMIDADE]\*\* — só a passada de conformidade acadêmica



A compilação LaTeX/PDF não foi verificada por ausência de toolchain completo no ambiente de revisão; itens que dependem de render estão marcados como não verificados.



\---



\## 1. Sumário executivo



O trabalho está bem estruturado e o código roda de ponta a ponta sem erros, produzindo os resultados que o texto descreve. A combinação de teste de diferença (principal) com teste de equivalência TOST (complementar) é sólida e responde ao enunciado. Os problemas concentram-se em quatro frentes: (i) um achado metodológico que afeta a conclusão principal, (ii) deliverables exigidos pelo enunciado que são calculados mas não aparecem no PDF, (iii) conformidade com o template e com convenções de escrita científica, e (iv) polimento textual.



Situação: não pronto para entrega. Bloqueiam os itens A, 1, 2, 3 e 4.



\---



\## 2. Bloqueiam a entrega



***### \[AMBOS] \[S4] A. Inconsistência metodológica no TOST sob não normalidade***

***Confirmado por execução: o grupo feminino de 2017 rejeita normalidade (Shapiro-Wilk p = 0,037 < 0,05). Por isso `testar\_diferenca()` corretamente troca para Mann-Whitney no feminino, mas `testar\_equivalencia()` usa `t.test()` em todos os casos, inclusive nesse mesmo grupo. Para o mesmo par de dados, o relatório usa um teste não-paramétrico para a diferença e um paramétrico (que assume normalidade) para a equivalência. Como toda a conclusão sobre o feminino repousa nesse TOST, o ponto é material. Os dois relatórios chegaram a ele de forma independente.***



***Ação (escolher um caminho, não deixar solto):***

***- \*\*Documentar como limitação:\*\* registrar no texto que o TOST foi aplicado via teste t mesmo onde a normalidade não pôde ser confirmada, com a justificativa de que, com n = 4, uma variante não-paramétrica de equivalência teria poder útil nulo — tornando a escolha consciente e defensável.***

***- \*\*Ou corrigir:\*\* implementar um TOST por bootstrap/permutação para os casos de não normalidade.***



***### \[AMBOS] \[S3] 1. Tamanho de efeito e intervalo de confiança não aparecem no PDF***

***`testar\_diferenca()` retorna `estimativa` e `ic`, mas a `tabela\_resultados` só mostra p-valores e n. O enunciado exige "Estimação do tamanho do efeito e do intervalo de confiança na grandeza de interesse". Ação: incluir estimativa e IC na tabela, com uma frase interpretando a magnitude.***



***Cuidado ao surfacear o IC (ver achado E abaixo): o IC do Welch (masculino) é de diferença de médias; o do Mann-Whitney (feminino) é de diferença de localização (Hodges-Lehmann). São grandezas distintas e não devem ir na mesma coluna sem nota.***



\### \[AMBOS] \[S3] 2. Premissas (Shapiro-Wilk) computadas mas nunca discutidas

Os p-valores de Shapiro-Wilk decidem qual teste roda, mas não aparecem na saída renderizada. O enunciado exige "Verificação e discussão das premissas do teste". Ação: reportar os dois p-valores por comparação e justificar em uma frase o ramo escolhido (Welch ou Mann-Whitney).



\### \[AMBOS] \[S3] 3. Estrutura de seções divergente do template

Dois aspectos:

***- \*\*Hierarquia (conformidade):\*\* "Conclusões e Recomendações" está como `###` (subseção de "Análise Estatística") e "Declaração de uso de IA" como `###` (sob "Referências"). Ambas deveriam ser `##`, no mesmo nível das demais seções. Nota: a justificativa de "regra de subdivisão mínima de duas subseções" citada na passada de conformidade é frágil aqui — o problema real é apenas o nível hierárquico; usar a correção, mas não colar essa justificativa sem conferir.***

\- \*\*Seções ausentes/fundidas (técnico):\*\* faltam os cabeçalhos "Descrição da Coleta de Dados", "Verificação das Premissas do Modelo" e uma "Análise Exploratória dos Dados" autônoma — o histograma está embutido em "Desenho do Experimento". "Obediência ao formato determinado" é critério de avaliação. Ação: restaurar os cabeçalhos e mover o histograma para sua própria seção.



***### \[TÉCNICO] \[S4] 4. Declaração de uso de IA***

***A declaração afirma uso do Google Gemini para redação e estruturação em LaTeX. É um ponto de exatidão factual sobre uma declaração formal; cabe ao grupo confirmar se corresponde às ferramentas efetivamente usadas antes da entrega. Ação: verificar e corrigir conforme o que de fato foi utilizado.***



\---



\## 3. Achados conceituais e metodológicos



\### \[TÉCNICO] \[S3] B. Desvio-padrão agrupado válido apenas no caso balanceado

`sd\_pool = sqrt((var(x) + var(y)) / 2)` é média simples das variâncias, que só coincide com o pooled correto quando os grupos têm o mesmo n. No masculino (21 e 21) está correto; no feminino (7 e 4) não é o pooled ponderado por graus de liberdade. Como o n ideal do feminino deriva desse número, herda o desvio. Não muda a conclusão qualitativa. Ação: usar a fórmula ponderada por graus de liberdade ou registrar a aproximação.



\### \[TÉCNICO] \[S2] C. "IMC permaneceu estável" (masculino) omite onde a média está

Médias masculinas: 24,94 (2016) e 24,29 (2017), ambas a menos de um ponto do corte de sobrepeso da OMS (25). Há equivalência estatística, mas "estável no topo da faixa normal, quase em sobrepeso" difere de "estável e saudável", dado o enquadramento do IMC como proxy de estilo de vida. Ação: situar em uma frase onde essa média estável se encontra.



\### \[TÉCNICO] \[S2] D. Média feminina de 2017 abaixo do corte de peso normal

Pela própria tabela de classificação do documento, a média feminina de 2017 (18,45) cai em "abaixo do peso" (< 18,5). O relatório trata o feminino só como problema de amostra e não menciona isso. Pode ser ruído (n = 4), mas é o tipo de sinal que o estudo diz procurar. Ação: registrar ao menos uma observação exploratória.



\### \[TÉCNICO] \[S2] E. Natureza do IC do Mann-Whitney

`wilcox.test(conf.int = TRUE)` retorna IC da diferença de localização (Hodges-Lehmann), não de médias. Relevante ao resolver o item 1 para não misturar grandezas na coluna de IC. Ação: distinguir em nota.



***### \[TÉCNICO] \[S2] F. Recomendação de n do TOST assume diferença verdadeira zero***

***`calcular\_n\_equivalencia()` simula com `rnorm(mean = 0)`. O poder do TOST é máximo em diferença zero e cai perto de ±δ\*, então o n obtido é próximo de melhor caso, não de suficiente garantido. As Recomendações apresentam "n ≈ 8" como o número, não como limite inferior otimista. Ação: uma cláusula divulgando a suposição.***



***### \[CONFORMIDADE] \[S3] G. Interpretação do grupo feminino extrapola os dados***

***O texto diz que "nem diferença nem equivalência" é evidência \*\*direta\*\* de falta de poder. Não é direta: é consistente com falta de poder, mas também poderia ocorrer com efeito real próximo de δ\*. Ação: trocar "fornece evidência direta de" por "é consistente com" uma limitação de poder.***



\### \[CONFORMIDADE] \[S2] H. Coluna Idade coletada e nunca usada (método sem resultado)

A padronização inclui "Idade" e o arquivo de 2017 a contém, mas ela nunca é usada nem discutida. Ação: ou remover da tabela de layout e da narrativa, ou mencioná-la como variável descritiva disponível fora do escopo atual.



***### \[CONFORMIDADE] \[S2] I. Descrição de H1 imprecisa no framework frequentista***

***"H1 ... é a hipótese que o teste procura evidências para aceitar" sugere que o teste busca confirmar H1. No framework usado no resto do documento, rejeita-se ou não se rejeita H0; não se "aceita" H1. Ação: reformular para a linguagem de rejeição de H0.***



\### \[TÉCNICO] \[S1] J. n ideal assume design balanceado

`power.t.test` retorna um único n assumindo n1 = n2. No feminino real (7 vs 4) o desenho é desbalanceado. Ação: uma cláusula explicitando a suposição.



\---



\## 4. Conformidade acadêmica



\### \[CONFORMIDADE] \[S2] K. Objetivos específicos não enunciados

A Introdução enuncia só o objetivo geral (comparar médias), mas o estudo faz três análises (diferença, equivalência, poder). Ação: enunciar objetivos específicos (i, ii, iii) na Introdução, para rastreabilidade.



\### \[CONFORMIDADE] \[S2] L. Promessa da Introdução sem lastro no corpo

A Introdução apresenta o IMC como proxy de estilo de vida, e a Conclusão retoma "métricas complementares", mas nenhuma seção do corpo discute as limitações do IMC como proxy. Ação: um parágrafo breve na discussão sobre essas limitações, dando lastro à recomendação final.



\### \[CONFORMIDADE] \[S1] M. Siglas não definidas na primeira ocorrência

\- TOST nunca expandido: definir como "teste de equivalência TOST (do inglês \*Two One-Sided Tests\*)".

\- ENGSIS: "graduação em Engenharia de Sistemas (ENGSIS)".

\- CSV: "arquivos CSV (do inglês \*Comma-Separated Values\*)".

\- OMS: como não é reutilizada como sigla corrente, manter apenas o nome por extenso.



\### \[CONFORMIDADE] \[S1] N. Figura, tabela e equação órfãs

\- Histograma (Figura 1): tem legenda via `fig.cap`, mas o texto não o referencia por número. Referenciar: "A Figura 1 apresenta...".

\- Tabela de resultados: sem título e não referenciada. Adicionar `caption =` ao `kable` e referenciar como "A Tabela 1...".

\- Equação do IMC: não numerada nem referenciada. Numerar (se o template suportar) ou ao menos mencionar no texto.



\### \[CONFORMIDADE] \[S1] O. Símbolo p\_TOST usado sem definição formal

Derivável do código (`max(p1, p2)`), mas o manuscrito não o define. Ação: definir uma vez, junto às equações do TOST.



\---



\## 5. Reprodutibilidade e governança dos dados



\### \[CONFORMIDADE] \[S2] P. Proveniência dos dados não descrita (FAIR)

Os arquivos são carregados por caminho relativo, mas o texto não informa origem, data de coleta, autoria ou se são públicos. Ação: um parágrafo de proveniência (fornecidos pela disciplina, referentes às turmas 2016-2 e 2017-2, autopreenchidos pelos alunos, etc.).



\### \[CONFORMIDADE] \[S1] Q. Sem registro de versões (sessionInfo)

O `setup` instala pacotes mas não registra versões de R e pacotes. Ação: adicionar um chunk final com `sessionInfo()`.



\### \[CONFORMIDADE] \[S1] R. Parâmetro n\_rep da simulação não justificado

`n\_rep = 2000` aparece só como default. A semente (42) está bem definida, o que é positivo. Ação: mencionar no texto o número de réplicas de Monte Carlo e a semente.



\---



\## 6. Polimento textual



\### \[CONFORMIDADE] \[S1] S. Estilo científico

\- "Nesse estudo" -> "Neste estudo" (este para o próprio trabalho).

\- "temos como objetivo" -> forma impessoal ("o objetivo deste estudo é").

\- Tempo futuro na Introdução ("serão aplicados") -> presente do indicativo ("são aplicados").

\- "(variável proxy)" sem itálico -> "(variável \*proxy\*)".

\- "de forma similar a normal" -> crase: "similar à normal".

\- "fortemente influenciados" -> "influenciados pelo tamanho amostral disponível".

\- "abaixo" -> "a seguir" (registro acadêmico).

\- Parágrafo do histograma "consistente com o Shapiro-Wilk já calculado": impreciso, pois o Shapiro roda depois. Atenção: a redação sugerida na passada de conformidade ("que será avaliado formalmente adiante") reintroduz o problema de olhar os dados antes de fechar o método. Melhor remover a antecipação e deixar o histograma como exploração pura, sem prometer o teste que vem depois.

\- Frase final do §2 da Introdução sem ponto final.



***### \[AMBOS] \[S0-S1] T. Erros de digitação***

***selcionar -> selecionar; proseguimos -> prosseguimos; codigo -> código; inciada -> iniciada; eta seguinte -> etapa seguinte; proxima secção -> próxima seção; incio -> início; ao ao teste -> ao teste; distribuiçãos -> distribuições. Observação: "secção" é grafia europeia; o resto do documento usa a forma brasileira "seção". Padronizar.***



\### \[TÉCNICO] \[S1-S2] U. Comandos LaTeX não verificados por compilação

`\\hyperlink`/`\\hypertarget` (link da referência da OMS), `\\qquad`, `\\vspace`. São padrão e devem funcionar, mas este arquivo já teve duas falhas reais de compilação (um caractere Unicode e uma URL corrompida). Ação: rodar `rmarkdown::render()` completo e confirmar que o link resolve.



\---



\## 7. Matrizes de rastreabilidade



\### Objetivos



| Objetivo | Método | Resultado | Retomado na Conclusão | Status |

|---|---|---|---|---|

| Comparar IMC médio 2016-2 vs 2017-2 (geral) | Desenho do Experimento | Tabela de resultados | Sim | Fechado |

| Testar diferença de médias | testar\_diferenca | p=0,59 (M); p=0,073 (F) | Sim | Fechado |

| Testar equivalência (TOST) | testar\_equivalencia | p\_TOST=0,019 (M); 0,31 (F) | Sim | Fechado |

| Avaliar adequação amostral | calcular\_n\_\* | n ideal vs n real | Sim | Fechado |



Os três últimos são implícitos: a Introdução não os enuncia como objetivos específicos (ver achado K).



\### Hipóteses



| Hipótese | Testada em | Resultado | Status |

|---|---|---|---|

| H0 (M): mu1 = mu2 | Welch | Não rejeitada (p=0,59) | Fechado |

| H0 (F): mu1 = mu2 | Mann-Whitney (normalidade falha) | Não rejeitada (p=0,073) | Fechado |

| TOST (M): equivalência | Dois t unilaterais | Confirmada (p=0,019) | Fechado |

| TOST (F): equivalência | Dois t unilaterais | Não confirmada (p=0,31) | Aberto — ver achado A |



\---



\## 8. Próximos passos



Ordem sugerida antes da entrega: (1) resolver o achado A escolhendo documentar ou corrigir; (2) surfacear efeito e IC, com atenção ao achado E; (3) surfacear as premissas de Shapiro-Wilk; (4) restaurar cabeçalhos do template e realocar o histograma; (5) resolver a declaração de IA com o grupo. Em seguida, os achados de conceito (B, C, D, F, G, I) e a camada de conformidade/estilo/reprodutibilidade (K a T). Por fim, rodar um `rmarkdown::render()` real e confirmar o link da referência (U).



Contagem por severidade: ***S4 x2 (A, 4)***; S3 x4 (1, 2, 3, B, G — G é S3 de conteúdo); S2 x9; S1 x9; S0-S1 x1.

