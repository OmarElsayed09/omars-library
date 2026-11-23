# Bilingual lecture evaluation

## What was measured

The corpus consists of the first three lecture transcripts from Gilbert Strang's
[18.06 Linear Algebra, MIT OpenCourseWare, Spring 2010](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
The download script retrieves the original PDFs directly from MIT.
[Course material terms](https://ocw.mit.edu/pages/privacy-and-terms-of-use/).

The [question set](lecture_cases.json) contains Egyptian Arabic, standard Arabic
and English. Each item names a selected lecture, an expected answer, and
one or more reference pages. Five questions ask for information absent from
the selected transcript. One conversation has a preceding question. The
development split has 15 questions and the held-out split has 6.

All questions went through the browser's streaming chat endpoint, including
routing, embedding, passage retrieval and generation. Each case starts a new
chat. Runs use a fresh Chroma index and chat database. Models were
`bge-m3` and `qwen2.5:3b`, with the cross-encoder disabled. The controlled
comparison changes only the query-translation setting. The latency measurement
covers the whole chat turn, including generation, on one local machine.

## Results

| Run | Reference page in top 4, answerable questions | Median time per turn |
| --- | --- | --- |
| Development, original query | 8/12 | 0.90 s |
| Development, bilingual query | 9/12 | 1.25 s |
| Held-out, bilingual query | 2/4 | 0.87 s |

Page matches use the reference pages specified before running the comparison.
Other pages can contain equivalent evidence. For example, page 3 of lecture 1
also mentions linear combinations but is outside that question's original
reference set. Counting it would bring the original-query development score
to 9/12 and leave the bilingual score at 9/12. The small difference above
should not be interpreted as a reliable overall quality gain.

For the Arabic question about a column times a row, the original query
returned pages 8, 8, 10 and 6. The translated query returned 3, 4, 4 and 5,
including the explanation on page 4. The answer in this run was just a
citation. In the opposite direction, translation lost page 12 for the
question about reversing an elementary row operation.

## Answer review

### Chat model and language-check update

A later [21-question run with the 7.6B model](reports/lectures-quality-7b.json)
tested revised answer instructions and the draft-language check on the original
lecture index. All 21 final responses were free of unexpected Chinese script.
The labeled page was retrieved for 11/16 answerable questions; median turn time
was 26.35 seconds on this machine. For example, the identity-matrix question
now has an English answer, while the outer-product answer has a correct worked
example but reverses the row and column definitions in the explanation.
One unanswerable question triggered the generic generation fallback instead
of an evidence-based abstention. This is a generation comparison; newly
rebuilt indexes also use the updated chunk-boundary algorithm.

### Original 3B run

Each saved answer was reviewed against its reference. An answerable item
passes if it states the requested fact intelligibly in the question's
language, allows technical terms in English, and has no material contradiction.
An abstention passes if it clearly declines to supply the missing fact.
The abstention score counts that decision regardless of response language.
These qualitative labels are a single assistant review, not independent
human annotations. They apply to the saved model outputs.

| Bilingual run | Correct answers | Correct abstentions |
| --- | --- | --- |
| Development | 4/12 | 3/3 |
| Held-out | 2/4 | 2/2 |
| Total | 6/16 | 5/5 |

In the order of [the saved run](reports/lectures-bilingual.json), answerable
cases 2, 3, 6, 7, 8 and 12 pass. Cases 5, 11, 14, 18 and 21 are correct
abstentions. The other ten are incorrect or inadequate: case 1 contradicts
the speaker's name in its own explanation, case 10 cites an unrelated
explanation of back substitution, case 16 returns only a reference, and
case 17 switches languages. Ranking a reference page does not guarantee
that the model used the evidence correctly.

## Reproduce

From the project root with the models installed:

```powershell
.\evaluation\fetch_lectures.ps1
$env:BILINGUAL_QUERY = "false"
.\.venv\Scripts\python.exe evaluate_lectures.py --split dev --output evaluation/reports/my-control.json
$env:BILINGUAL_QUERY = "true"
.\.venv\Scripts\python.exe evaluate_lectures.py --split all --output evaluation/reports/my-bilingual.json
```

Inspect [control answers](reports/lectures-original-query.json) and
[bilingual answers](reports/lectures-bilingual.json) for per-question times,
pages, references and complete generated responses. The runner recomputes
page hits and timing automatically. Review new answers separately because
model wording and run times vary.
