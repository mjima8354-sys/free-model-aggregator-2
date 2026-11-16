# Live Free Model Aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/mjima8354-sys/free-model-aggregator-2/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/mjima8354-sys/free-model-aggregator-2/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One deterministic server that fronts OpenRouter's free models behind two client wires: the OpenAI Chat Completions API and the Anthropic Messages API, each in JSON and in server-sent events. Entered on make.fun's "The best live free-model aggregator" bounty.

## What it does

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`: the sealed live catalog of OpenRouter free models with the house's quality ratings, and the six-prompt manifest) and derives its catalog from it alone — no probing, no network guesswork.
- Registers its port and catalog with the contest env (`POST /register`), serves `GET /v1/models` in both wires, `POST /v1/chat/completions` and `POST /v1/messages` (JSON + SSE), and polls `GET /work`, exiting 0 when the env says done.
- Routes every prompt to the highest-rated free model that meets its requirements (tools, vision, JSON mode), dialled through the grader's gateway Chat Completions route on the trial's key; Anthropic-shaped requests are translated to a Chat Completions body and the answer translated back, with usage and stop reasons whole.
- Puts the dialled `openrouter/...` id back on every answer and every streamed chunk, so the `model` field always names the model that really answered.
- Fails over on every refusal: a 429/5xx/402 moves the prompt to the next qualified candidate, with cooldowns and one re-probe, and reports recovered refusals in the optional `x-makefun-upstream-failures` trail header.
- Is deterministic by construction: no wall clock, no unseeded randomness; a replay with the same gateway answers leaves the same state export.

## Run it

```sh
python3 main.py --list        # print the catalog and exit
python3 main.py               # run inside a trial (MAKEFUN_ENV_URL / MAKEFUN_PROMPT_FILE set)
```

Python 3.13, standard library only. Inside a trial it listens on `0.0.0.0` at a free port it picks and registers with the env.

## License

MIT — see [LICENSE](LICENSE).
