# Next-token prediction example

Story (first 25 tokens shown):

> once upon a time, there was a little girl named lily. she loved to play with her dolls and draw pictures. one ...

| step | input token | target (next) token |
|---|---|---|
| 0 | `<sos>` | `once` |
| 1 | `once` | `upon` |
| 2 | `upon` | `a` |
| 3 | `a` | `time` |
| 4 | `time` | `,` |
| 5 | `,` | `there` |
| 6 | `there` | `was` |
| 7 | `was` | `a` |
| 8 | `a` | `little` |
| 9 | `little` | `girl` |
| 10 | `girl` | `named` |
| 11 | `named` | `lily` |
| ... | ... | ... |
| 181 | `.` | `<eos>` |

Full id sequence length: 183 (= `<sos>` + 181 words/punctuation + `<eos>`).
