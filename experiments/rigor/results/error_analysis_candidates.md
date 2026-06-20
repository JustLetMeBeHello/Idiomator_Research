# IdiomBERT Table 9 — Error Analysis Candidates

Auto-sampled heuristically. **Review each row: keep, edit, or drop.**
Target: 20–40 rows total, ~5 per category, ≥3 per language.
Copy accepted rows into Table 9 in the docx.

> Category codes: **FN-I** = Idiomatic → literal (FN)  |  **FP-L** = Literal → idiomatic (FP)  |  **SB** = Span boundary  |  **SM** = Span missed entirely  |  **SC** = Sense confusion  |  **CL** = Cross-lingual transfer  |  **AM** = Ambiguous gold

## FN-I — Idiomatic → literal (FN). Figurative use labelled literal.

| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |
|---|-----|------|-----------------------|------|------|-------|
| 1 | E | En | If she **flips her lid** over the broken vase, I’ll have to explain it was an accident. | idio | lite | overlap=0.89; pred_span="flips her lid over"; sense=2 |
| 2 | E | En | She designed the user interface to be **simplicity itself**, so even beginners could navigate it wit… | idio | lite | overlap=0.67; pred_span="user interface to be simp…" |
| 3 | E | Sp | **Al primer golpe de vista**, la casa parecía abandonada, pero al acercarnos descubrimos que alguien… | idio | lite | [TODO: add notes] |
| 4 | E | Sp | Después de una semana estresante, decidí **tomar el sol** y relajarme en el jardín. | idio | lite | sense=2 |
| 5 | E | En | Did you see that the chess match **tied in** the final round? Neither player could gain the upper ha… | idio | lite | pred_span="see that the chess match"; sense=3 |
| 6 | E | Sp | En el mercado había frutas **a tutiplén**, suficientes para abastecer a toda la ciudad. | idio | lite | [TODO: add notes] |

## FP-L — Literal → idiomatic (FP). Literal use labelled idiomatic.

| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |
|---|-----|------|-----------------------|------|------|-------|
| 7 | G | En | The pressure was **raised as the stakes** became higher in the final round of the championship. | lite | idio | overlap=0.30; pred_span="stakes became higher"; sense=3 |
| 8 | G | En | The firefighters spent all night **fighting fires** caused by the dry lightning storm in the forest … | lite | idio | pred_span="dry lightning storm" |
| 9 | G | Sp | No sentí ningún **tilín** en los dedos pese al frío intenso que hacía esa mañana. | lite | idio | pred_span="en los dedos"; sense=2 |
| 10 | G | Sp | Para el experimento, **sacaron lo mejor del** mineral para analizar su composición. | lite | idio | [TODO: add notes] |
| 11 | G | Hi | **मरे बिना स्वर्ग नहीं मिलता**, इसलिए हमें अपने कर्म सुधारने चाहिए। | lite | idio | overlap=0.92; pred_span="बिना स्वर्ग नहीं मिलता" |
| 12 | G | Hi | कहा जाता है कि **मरे बिना स्वर्ग नहीं मिलता**, इसका अर्थ है कि मृत्यु के बाद ही मोक्ष संभव है। | lite | idio | [TODO: add notes] |

## SB — Span boundary. Idiom detected, span off by ≥1 token.

| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |
|---|-----|------|-----------------------|------|------|-------|
| 13 | E | En | When the radical group started managing the town council, locals joked that the **lunatics had taken… | idio | idio | overlap=0.40; pred_span="taken over" |
| 14 | G | En | **For your particular** benefit, I’ve added detailed instructions to the manual. | idio | idio | overlap=0.27; pred_span="For" |
| 15 | E | Sp | La abuela guarda las joyas de la familia **a buen recaudo** en un lugar secreto. | idio | idio | overlap=0.53; pred_span="guarda las joyas de la fa…" |
| 16 | E | Sp | Si hubiéramos escuchado a Leonardo, que era **adelantado a su época**, la tecnología estaría mucho m… | idio | idio | overlap=0.57; pred_span="escuchado a Leonardo, que…" |
| 17 | G | En | I know this isn’t my area of expertise, but here’s my **two cents** on the matter. | idio | idio | overlap=0.56; pred_span="two cents on the matter" |
| 18 | G | En | When the radical group started managing the town council, locals joked that the **lunatics had taken… | idio | idio | overlap=0.45; pred_span="taken over" |

## SM — Span missed entirely. No span produced for detected idiom.

| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |
|---|-----|------|-----------------------|------|------|-------|
| 19 | G | En | The detective found more strange clues at the scene and exclaimed, 'This case is getting **curiouser… | idio | idio | pred_span="strange clues" |
| 20 | G | En | If you think you can cheat on the test and get away with it, that's a **big fat** mistake. | idio | idio | pred_span="get away"; sense=2 |
| 21 | G | Sp | En la reunión de juego, dejé que Marta **ganara la partida** para animarla un poco. | idio | idio | pred_span="de juego"; sense=4 |
| 22 | G | Sp | Después de tanto esfuerzo, su éxito **está bien empleado** y todos lo reconocen. | idio | idio | sense=2 |
| 23 | E | Hi | काम पूरा हो गया, **फिर से** बात ये है कि उसे समय पर करना जरूरी था। | idio | idio | pred_span="बात"; sense=2 |
| 24 | G | Hi | माँ ने बच्चे को डांट कर **ठिकाने लगा दिया** कि वह ध्यान से पढ़ाई करे। | idio | idio | pred_span="डांट कर"; sense=4 |

## SC — Sense confusion. Multi-sense idiom; wrong sense (candidate).

| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |
|---|-----|------|-----------------------|------|------|-------|
| 25 | E | En | Managers should encourage teams to **make haste slowly** to balance efficiency and accuracy. | idio | idio | overlap=0.86; pred_span="make haste"; sense=2 |
| 26 | G | En | They decided to **run on** more units of the product due to unexpectedly high demand. | idio | idio | overlap=0.71; pred_span="run on more"; sense=3 |
| 27 | G | Sp | Cuando Juan decidió hacerse a **la mar**, sabía que enfrentaría muchos desafíos en su viaje. | idio | idio | overlap=0.55; pred_span="hacerse a la mar"; sense=2 |
| 28 | G | Sp | **Me hace falta** mi abuela cuando cocinaba sus recetas tradicionales para toda la familia. | idio | idio | overlap=0.87; pred_span="hace falta"; sense=4 |
| 29 | G | Hi | अगर तुम्हें कभी कोई **ईद का चाँद** मिले, तो उसकी कद्र करना। | idio | idio | overlap=0.80; pred_span="ईद का चाँद मिले"; sense=2 |
| 30 | G | Hi | वह हमेशा **जलते घर में हाथ सेंकता** रहता है और अपने फायदे के लिए झगड़ों में कूद पड़ता है। | idio | idio | overlap=0.31; pred_span="जलते"; sense=2 |

## CL — Cross-lingual transfer. Error in low-resource lang (HI/TE/ID).

| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |
|---|-----|------|-----------------------|------|------|-------|
| 31 | C | Hi | बच्चों ने **घड़ियाली आँसू** देखकर उसकी पीड़ा महसूस की। | lite | ? | [TODO: add notes] |
| 32 | C | Hi | धार्मिक ग्रंथ कहते हैं कि **मरे बिना स्वर्ग नहीं मिलता**। | lite | ? | pred_span="मरे बिना स्वर्ग नहीं मिलत…" |
| 33 | C | Te | **చాపక్రిందినీరు** చాలా శుభ్రమైనది, అందరూ ఆశ్చర్యపోయారు. | lite | ? | [TODO: add notes] |
| 34 | C | Te | ఆ కంపెనీ వాటా ధరలు **కొండెక్కు**, ఇన్వెస్టర్లు సంతోషించారు. | idio | ? | sense=2 |
| 35 | C | Te | రామ్ ఎప్పటికీ **చంకనాకిపోతోందని** నాకు అనిపిస్తుంది, కానీ అతను ఎందుకు అలా మాట్లాడాడో అర్థం కాలేదు. | idio | ? | [TODO: add notes] |
| 36 | C | Te | ఆమె **కడుపు చేతబట్టుకొని** తృప్తిగా నవ్వింది. | lite | ? | [TODO: add notes] |

## AM — Ambiguous gold. Borderline annotation; flagged for re-annotation.

| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |
|---|-----|------|-----------------------|------|------|-------|
| 37 | G | En | She said she was going **round the bend** with all the noise from the construction site next door. | idio | idio | overlap=0.82; pred_span="going round the bend" |
| 38 | G | En | After working nonstop for six months, Maria finally took some **time off** to recharge. | idio | idio | overlap=0.62; pred_span="took some time off" |
| 39 | E | Sp | Por más que intentó no **asomar las narices** en la reunión, finalmente lo vieron llegar. | idio | idio | overlap=0.71; pred_span="Por más que intentó no as…" |
| 40 | G | Sp | El salario de Ana **va a la par** con su experiencia y responsabilidades. | idio | idio | overlap=0.84; pred_span="a la par" |
| 41 | G | Sp | Si te **campaneas** mucho, terminarás con problemas en la escuela. | idio | idio | overlap=0.67; pred_span="te campaneas mucho" |
| 42 | E | Sp | Nosotros **vamos a la par** en el proyecto, así que debemos coordinarnos para no atrasarnos. | idio | idio | overlap=0.75; pred_span="a la par" |

---

Total candidates: 42
Final Table 9 target: 20–40 rows, ~5 per category, ≥3 per language.

After selecting rows: paste into Table 9 in IdiomBERT_Submission_Ready_v8.docx §9.