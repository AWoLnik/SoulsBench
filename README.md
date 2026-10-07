# SoulsBench

> [!WARNING]
> **This project is entirely vibe coded.** Every line was written by an AI coding agent (Claude
> Code) from natural-language instructions, with a human steering, watching the game and pressing
> "load game". It has a test suite and was run for many hours against the real game, but nobody
> has carefully reviewed the code. Expect rough edges, read it before you trust it, and use a
> throwaway save (see below).

A Dark Souls III **realtime combat benchmark for decision models**. Every model plays Iudex Gundyr
on the live game through the same harness, the same `CombatState`, the same 16 actions and the
same 200 ms decision budget, while the game keeps running. Models only differ in the adapter that
turns a state into a decision.

## Results

![Win rate over 50 realtime boss fights against Iudex Gundyr](docs/full50_win_rates.png)

| model | wins / 50 | avg boss damage (of 1037) | p50 latency | cost per fight |
|---|---|---|---|---|
| OpenAI `gpt-6-luna` | **32%** | 701 | 97 ms | $0.047 |
| TypeSafe Jev | 24% | 698 | 147 ms | $0.024 |
| Perplexity `pplx-decider-v1-27b` | 12% | 484 | 158 ms | $0.010 |
| Kev 0.8B (local) | 6% | 434 | 20 ms | local |
| Laya ~210M (local) | 0% | 361 | 30 ms | local |

Full fight (both phases), seeds 0–49, one request per decision, the same prompt for every model.
Details, confidence intervals and caveats (the prompt was tuned on Jev; Perplexity's API degraded
for part of the run): [docs/results_full50.md](docs/results_full50.md).

## How it works

```
 WSL (Linux)                                   Windows (native)
 ┌──────────────────────────────┐   HTTP/JSON  ┌──────────────────────────┐   memory r/w,   ┌──────────────┐
 │ soulsbench.bench / .run      │ ───────────► │ soulsbench.windows_bridge │   keystrokes,   │ DarkSoulsIII │
 │  harness: reset → decide →   │ ◄─────────── │  SoulsGym IudexEnv       │ ◄─────────────► │  .exe 1.15.2 │
 │   act (realtime) on arrival  │  :47800      │  (+ SoulsBench subclass)  │   speedhack     └──────────────┘
 │  adapters: openai, jev, pplx,│              │  OBS recording (optional) │
 │   kev, laya, rules, http, …  │              └──────────────────────────┘
 └──────────────────────────────┘
```

The game itself never runs under Wine. [SoulsGym](https://github.com/amacati/SoulsGym) drives the real
Windows process: it reads and writes game memory, sends keystrokes, and controls game speed through
an injected speedhack DLL. In realtime mode (the default) the game runs continuously during a fight;
it only pauses between episodes.

## Requirements

* **An owned Steam copy of Dark Souls III, version 1.15.2, with both DLCs installed.** SoulsGym's
  memory offsets are for exactly that build. Nothing here works without the game, and nothing here
  ships game files.
* Windows 10/11 with WSL2. The repo lives in the WSL filesystem (e.g. `~/Sources/SoulsBench`), not
  under `/mnt/c`.
* A single **en-US keyboard layout** on Windows. SoulsGym's warning: a second installed layout can
  swallow keystrokes, even when it is not active.

> **Your saves.** SoulsGym overwrites the loaded character's stats, last bonfire, boss flags and
> play time. Use a throwaway **Knight** created for this, in **offline** mode. The bridge refuses to
> start on any character that is not a fresh SL9 Knight (`--allow-any-save` overrides this), and it
> copies `%APPDATA%\DarkSoulsIII` to `%LOCALAPPDATA%\soulsbench\save_backups\<timestamp>` every time
> it starts. Delete the SoulsBench character before going online again.

## 1. Windows bridge setup (once, from Windows)

From a Windows PowerShell (not from inside WSL):

```powershell
powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\<you>\Sources\SoulsBench\windows\setup_bridge.ps1
```

This creates everything under `%LOCALAPPDATA%\soulsbench`. It does not install a system Python or
touch PATH.

| piece | exact version | why |
|---|---|---|
| Python | **CPython 3.12.15, 64-bit** (uv-managed) in `%LOCALAPPDATA%\soulsbench\venv` | SoulsGym needs ≥ 3.11.2; 3.12 is what this bridge is tested with |
| SoulsGym | **1.2.0 @ [`29c94d8`](https://github.com/amacati/SoulsGym/tree/29c94d89ac8a47846c619019296d0ba55c6e80d5)** (GitHub archive, no git needed) | targets DS3 1.15.2, Rust speedhack, abi3 wheels. PyPI only has 1.1.0, whose screen-capture `.pyd` is built against `python311.dll`, so it requires exactly CPython 3.11 |
| everything else | pinned in [`windows/requirements-bridge.txt`](windows/requirements-bridge.txt) | e.g. pywin32 312, pymem 1.14.0, numpy 2.5.3, opencv-python 4.14 (capped < 5) |
| SoulsBench | editable install of this repo, over `\\wsl.localhost` | edits in WSL apply on the next bridge start (`-CopySource` makes a local copy instead) |

At the end the script runs `python -m soulsbench.windows_bridge doctor`. It checks the Python and
SoulsGym versions, the DS3 version and screen mode, and keyboard layouts.

**Networking: no firewall rule, no admin.** The bridge listens on Windows `127.0.0.1` only, so it is
never exposed and never triggers a firewall prompt. WSL2's default NAT networking cannot reach Windows'
loopback directly. The WSL client therefore starts a tiny relay in the bridge's own Windows Python
through WSL interop: one process per connection, about 1 ms per step
([`backends/relay.py`](soulsbench/backends/relay.py)). The client tries, in order:

1. `SOULSBENCH_BRIDGE_URL`;
2. `127.0.0.1` directly (works with WSL mirrored networking);
3. the relay;
4. the Windows host IP, which needs `--host <WSL adapter IP>` on the bridge plus the elevated
   [`windows/allow_wsl_firewall.ps1`](windows/allow_wsl_firewall.ps1).

## 2. Dark Souls III setup (once, in the game)

1. **System → Network → Launch Setting: Play Offline.** Restart the game if it asks.
2. **Settings → Graphics → Screen Mode: Windowed**, at any resolution. 1920×1080 is good for
   recording. Windowed is mandatory: SoulsGym re-focuses the game at every reset (bring to front,
   then click inside it), and exclusive fullscreen minimizes or mode-switches whenever focus moves.
   The bridge refuses to start on a fullscreen game. SoulsGym's own docs ask for 800×450 because
   its screen-capture class checks for that size; SoulsBench reads everything from memory and
   swaps that class for a focus-only one, so the size is free and nothing captures the screen.
3. **New Game → Knight.** Default appearance and gifts are fine. Play until you control the
   character in the Cemetery of Ash. Do not level up or change equipment.
4. **System → Input Device → Key Bindings (keyboard)**, exactly as below. Fields marked `---` must be
   explicitly *unset*.

| action | keyboard | mouse | used for |
|---|---|---|---|
| Run (forward / backward / left / right) | W / S / A / D | | move_*, roll_* direction |
| Dash / Backstep / Roll | Space | | roll_*, backstep |
| Jump | Space | | |
| Tilt camera up / down / left / right | J / K / O / P | | SoulsGym camera assist |
| Camera reset / Lock-on | Q | | toggle_lock, lock-on assist |
| Attack (right hand) | L | `---` | light_attack |
| Strong attack (right hand) | H | | heavy_attack |
| **Attack (left hand)** | **U** | `---` | **block** (Knight Shield). SoulsGym leaves this unset; SoulsBench needs it |
| Strong attack (left hand) | M | | parry (Knight Shield skill) |
| **Use item** | **R** | | **use_estus** |
| Interact | E | | fog gate (SoulsGym reset) |

5. Make sure the **Estus Flask** (not the Ashen Estus Flask) is the selected quick item.

## 3. Run the bridge (from Windows)

```powershell
powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\<you>\Sources\SoulsBench\windows\run_bridge.ps1
```

The bridge:

* attaches to `DarkSoulsIII.exe`, or launches it through Steam (`steam.exe -applaunch 374320`).
  `--no-launch` attaches only.
* backs up the saves (`--no-backup` skips this).
* serves on `127.0.0.1:47800` only (`--port`; `--host` adds bind addresses).
* logs to `%LOCALAPPDATA%\soulsbench\bridge.log`, readable from WSL under `/mnt/c/Users/<you>/AppData/Local/soulsbench/`.
* waits for you to **load the Knight**, then refuses a fullscreen game or a non-Knight character.
* reports `ready` on `/health`.

While episodes run the bridge owns the keyboard and the DS3 window. Do not touch them. Ctrl+C
restores SoulsGym's game flags and resumes the game, which respawns the character (SoulsGym's own
`close()`). The first reset of a fresh Knight also kills the character once, on purpose: SoulsGym
respawns it at the Cemetery of Ash bonfire and walks it through Gundyr's fog gate.

**Recording (OBS etc.).** Capture the DS3 window, e.g. Windowed 1920×1080 on a 1440p screen. The
game freezes for the length of each `decide()` call. With the built-in adapters this is about a
millisecond, so the video looks continuous. A remote model that thinks for 150 ms produces visible
stutter, which you can cut or speed-ramp in editing.

From WSL, the same thing is one command each. `start` opens `run_bridge.ps1` in its own Windows
console window, so it is still a native Windows process:

```bash
uv run python -m soulsbench.bridge start            # add -- --no-launch to only attach
uv run python -m soulsbench.bridge status           # exit code 0 once ready
uv run python -m soulsbench.bridge stop             # restore the game, exit the bridge
```

## 4. WSL client setup and one-command runs

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing
cd ~/Sources/SoulsBench && uv sync --extra dev
```

One command starts episodes against the live game (realtime by default):

```bash
uv run python -m soulsbench.run --backend souls_gym --scenario gundyr_full --adapter expert --seeds 3
uv run python -m soulsbench.run --backend souls_gym --scenario gundyr_full --adapter openai,jev --seeds 10
```

Each adapter gets `runs/<stamp>_<scenario>_<adapter>/` with:

| file | contents |
|---|---|
| `config.json` | scenario, SLA, seeds, bridge URL and health (versions) |
| `steps.jsonl` | one line per tick: requested/executed action, fallback status, decide latency, step RTT, HP, damage, anims (`--log-states` adds the full CombatState) |
| `episodes.jsonl` | per-episode metrics |
| `scorecard.json` / `scorecard.md` | means ± std over the seeds |

Several adapters also produce `runs/<stamp>_<scenario>_comparison.md`. To compare any runs later:
`uv run python -m soulsbench.report runs/`.

### The full benchmark (interleaved, resumable)

`soulsbench.bench` runs several models on the same seeds in rounds (seed 0 for every model, then
seed 1, …, adapter order rotated each round), so API conditions and game drift over a long night
hit every model alike. Everything goes to `runs/bench_<name>/`; re-running the same command
resumes, skipping (model, seed) pairs that already have a valid episode.

```bash
uv run python -m soulsbench.bench --name full50 --adapters openai,jev,pplx --seeds 50 --record-best
uv run python -m soulsbench.bench ... --skip pplx    # leave one out for now (an API is down); fill in later
scripts/run_full_bench.sh                            # what produced the results above
```

Safety rails: an episode that aborts (game or bridge error) is retried, and after 3 in a row the
bench stops; a fight in which more than 80% of decisions timed out is recorded as aborted (a
stalled server, not the model) and stops the bench. Local GPU models cannot share a 12 GB GPU with
DS3 and each other, so `run_full_bench.sh` runs each in its own pass with only its server loaded
(`scripts/serve_kev.sh`, `scripts/serve_laya.sh`).

### Recording fights (optional, OBS)

The Windows bridge can drive [OBS Studio](https://obsproject.com) over obs-websocket: set up a scene
with a **Game Capture** of DS3 plus desktop audio (NVENC keeps the encoding off the CPU), enable
*Tools → WebSocket Server Settings*, and put its password in `./obs_password.txt` (gitignored).
`--record-seeds 0-2` records those seeds for every model; `--record-best` records every fight and
keeps only each model's best (a win beats a loss, then HP left / boss damage), deleting the rest as
it goes. Videos land in `%USERPROFILE%\Videos\SoulsBench\<bench name>\`.

Post-ready media from a finished bench (fonts: [Cinzel](https://fonts.google.com/specimen/Cinzel)
and [Inter](https://fonts.google.com/specimen/Inter), SIL OFL, in one folder):

```bash
uv run --with matplotlib python scripts/win_rate_chart.py --fonts <font dir>        # the chart above
uv run --with pillow --with imageio-ffmpeg python scripts/make_clips.py --fonts <font dir>   # labelled 30 s clips
```

### Score card

| metric | definition |
|---|---|
| `win` | Gundyr killed (`gundyr_full`), or his phase-1 HP pool (1037) depleted (`gundyr_p1`), before the player dies or time runs out |
| `boss_hp_dealt` | damage dealt to that pool (also as a fraction) |
| `survival_s` | game seconds survived, capped at the scenario limit (300 s full fight, 180 s phase 1) |
| `player_hp_remaining` | real game HP at the end (0 on death) |
| latency p50 / p90 / p95 / p99 / max | wall-clock `decide()` latency over decisions with a real choice (more than one legal action); forced idle ticks are excluded so adapters that skip them are not flattered; plus a histogram |
| `timeout_rate` | decisions with no answer within the 200 ms budget |
| cost | API calls and input/output tokens as each provider reports them (warm-up calls and late answers included), priced at published rates (`soulsbench/pricing.py`): total, per fight, per win |
| also | over-budget, parse-failure, adapter-error and illegal-action rates; any-fallback-to-idle rate |

Reported numbers are means over **N ≥ 10 seeds**. Fewer seeds still work (`--seeds 3`), but the score
card flags them as smoke tests.

**Game RNG is not controllable.** A seed fixes the adapter's RNG, and the start pose in
`gundyr_p1_randstart`. Gundyr's AI, damage variance and the game's frame timing are not seeded, so the
same seed with the same adapter can play out differently. Compare means over seeds, never single
episodes.

## The contract (shared by every model)

### Loop and SLA

Realtime (default): `reset → decide(state, deadline) → act → decide(fresh state) → act …` until
win, death or time limit. Paused: `reset → decide → step → decide → step …`.

* **Decide budget 200 ms** of wall-clock per decision.
* **Realtime (the default): the game runs continuously and never waits for the model.** Each answer
  is applied the moment it arrives (`POST /act`: the bridge checks legality against the game as it
  is right then, presses the keys and returns the state read straight after). The next decision
  starts on that state immediately, so a model's latency is its reaction time and a faster model
  also acts more often. A timeout applies `idle` at the deadline. Actions are at least
  `--min-action-interval-ms` (50 ms) apart, which only ever slows instant scripted baselines. The
  bridge times animations with a background thread polling every ~4 ms, so the telegraph is
  accurate between actions.
* `--paused` is the old mode: SoulsGym's speedhack freezes the game while the model thinks, then one
  **control period** (250 ms) of game time plays per step. Score cards and comparison rows mark it
  `PAUSED` / `(paused)`; it is not comparable with realtime. Score cards written before realtime
  existed are read as paused.
* Realtime runs from 2026-10-04 to 10-06 used an earlier design that held each answer until the next
  250 ms tick boundary. `soulsbench.report` labels those cards `(tick-lagged rt)`; they are not
  comparable with current realtime results.
* **Timeout, parse failure, adapter error, over-budget, or an illegal action all become `idle`**,
  and each is counted.

### Discrete(16) → DS3 inputs

| id | action | keys (SoulsGym binding) | legal when |
|---|---|---|---|
| 0 | idle | none | always |
| 1–4 | move_forward/back/left/right | hold W / S / A / D | movement window open |
| 5–8 | roll_forward/back/left/right | direction + tap Space | roll window open, stamina > 0 |
| 9 | light_attack | tap L | attack window open, stamina > 0 |
| 10 | heavy_attack | tap H | attack window open, stamina > 0 |
| 11 | block | hold U (left-hand attack = guard) | movement window open |
| 12 | parry | tap M (Knight Shield skill) | attack window open, stamina > 0 |
| 13 | use_estus | tap R | attack window open, estus > 0 |
| 14 | backstep | tap Space with no direction | roll window open, stamina > 0 |
| 15 | toggle_lock | press Q | lock-on lost (see below) |

Directions are relative to the lock-on target: SoulsGym keeps the camera locked on Gundyr.
"Windows" are SoulsGym's measured per-animation input windows. Inside an animation the game would
queue an input and fire it later, so those inputs are masked instead. Unknown player animations
(e.g. drinking) are locked for their full length as read from memory. There is no camera action, and
SoulsGym re-locks a lost lock on its own. So `toggle_lock` is legal only to re-acquire a lost lock;
releasing the lock would just be undone on the next tick.

### CombatState (every tick; values illustrative)

```jsonc
{
  "schema_version": "soulsbench/v1", "episode_id": "…", "scenario": "gundyr_p1", "tick": 12, "t_s": 3.0,
  "player": {
    "hp": 380, "hp_max": 454, "stamina": 62, "stamina_max": 95,
    "poise": null, "poise_max": null,          // super-armor durability; null until a working pointer is found
    "estus": 3, "estus_max": 3,                // inventory (memory); null if unreadable
    "iframes": false, "parry_window": false,   // game flags (memory)
    "anim": "RollingMedium", "anim_id": 21, "anim_time_s": 0.31,
    "disabled": true,                          // locked in an animation
    "locked_on": true, "position": [x, y, z], "heading_rad": -2.77
  },
  "target": {
    "name": "Iudex Gundyr", "hp": 811, "hp_max": 1037, "boss_phase": 1,
    "position": [x, y, z], "heading_rad": 0.6,
    "telegraph": {
      "anim": "Attack3004", "anim_id": 4, "kind": "attack",   // attack | movement | stagger | unknown
      "progress": 0.42,                                       // anim time / anim length (memory)
      "anim_time_s": 0.9,
      "phase": "active",              // neutral | attack | windup | active | recovery | staggered | unknown
      "hitbox_active": true,          // null when this attack has no calibrated window
      "next_hit_in_s": 0.0,           // seconds until the next calibrated hit window; null if none
      "timing_source": "calibrated",  // calibrated | none | n/a
      "followup_p": 0.8,              // how often this attack chains straight into another one
      "distance": 2.1, "angle_off_deg": 12.0
    }
  },
  "history": [ { "tick": 11, "t_s": 2.75, "action": 9, "player_hp": 380, "target_hp": 811,
                 "player_anim": "AttackRightLight1", "target_anim": "Attack3004" } ],  // last 8
  "legal_actions_mask": [true, …16 booleans…]
}
```

**Telegraph honesty.** Without calibration, an attack reports `phase: "attack"` and `hitbox_active:
null`, never a guess. To calibrate, play some episodes (Idle and Random get hit a lot), then run
`uv run python -m soulsbench.calibrate runs/<run dirs>`. This writes observed hit windows per attack to
`soulsbench/windows_bridge/data/attack_windows.json`, which the bridge loads at start. On realtime
logs, `--hit-times` clusters precise hit times instead, giving multi-hit attacks one window per hit.

### Adapters

`CombatAdapter.reset(episode_id)`, `CombatAdapter.decide(state, deadline_ms) → Decision {discrete_id}`.

* `idle`: always 0.
* `random`: uniform over the legal actions, seeded per episode.
* `expert` (ScriptedExpert): drink estus below 40% HP, otherwise close to 2.5 m and light attack. It
  does not dodge or read telegraphs; it exists to show that the plumbing works and damage registers.
* `rules` / `rules:<ms>`: the tuned prompt's rules executed locally with no model (below), optionally
  delayed to a model's latency. It is the control for that prompt: if it wins as often as a model,
  the model's result is the rules'.
* `http` / `http:<name>`: any model behind a base URL (below).
* `openai`: OpenAI's Decisions API with `gpt-6-luna` (below).
* `jev`: Jev (TypeSafe AI System One) via its native API, `POST https://api.typesafe.ai/v1/systemone`
  with model `jev-latest`, one `choice` question over the actions. The key comes from
  `JEV_AI_API_KEY`, or from `./jev_api_key.txt` (gitignored).
* `kev` / `laya` / `systemone:<name>`: locally served models that speak the same `/v1/systemone`
  contract (defaults `127.0.0.1:8009` and `:8010`; override with `SOULSBENCH_<NAME>_BASE_URL`,
  `_MODEL`, `_API_KEY`). The results above used Laya's `typed-decisions` checkpoint:
  `systemone:laya-typed` with `SOULSBENCH_LAYA_TYPED_MODEL=typed-decisions`.

**Prompts.** All System One-style adapters (jev, openai, pplx, clef, kev, laya) share one encoding,
so a comparison is like for like. The default `tuned` prompt sends a few precomputed `tactical`
fields (the boss's `situation` (hit_incoming, attack_starting, string_continues, opening, neutral,
transforming), whether he is in sword or lunge range, whether HP is low and drinking is safe) and
asks for the first action whose condition holds; it was iterated on Jev. Append `@r5` for the
previous, plain-rules prompt or `@base` for the original untuned one (e.g. `openai@base`).
* `clef` / `clef-flash`: Cloudflare's Clef (27B) and Clef-flash (9B) on Workers AI, which use Jev's
  request format at `POST https://api.cloudflare.com/client/v4/accounts/<id>/ai/run/@cf/cloudflare/<model>`.
  The credentials are a Workers AI Read+Edit API token in `CLOUDFLARE_API_TOKEN` (or
  `./cloudflare_api_key.txt`) and the account ID in `CLOUDFLARE_ACCOUNT_ID` (or
  `./cloudflare_account_id.txt`).
* `pplx`: Perplexity's Decisions API with `pplx-decider-v1-27b`, `POST https://api.perplexity.ai/v1/decisions`,
  also in Jev's format. The key comes from `PERPLEXITY_API_KEY` or `./perplexity_api_key.txt`.

## Adding your own model

A model that speaks Jev's `/v1/systemone` contract (state + typed `choice` question) plugs in as
`systemone:<name>` with `SOULSBENCH_<NAME>_BASE_URL` (and optional `_MODEL`, `_API_KEY`), and gets the
same prompt as everything above. Anything else can implement the minimal SoulsBench `/decide`
protocol and needs only a base URL:

```bash
export SOULSBENCH_MYMODEL_BASE_URL=http://my-host:9000   # optional: SOULSBENCH_MYMODEL_API_KEY=… (Bearer)
uv run python -m soulsbench.run --backend souls_gym --scenario gundyr_full --adapter http:mymodel --seeds 10
# or a one-off:   --adapter http --http-base-url http://somewhere:9000
```

The model server implements:

* `POST {base}/decide`, with body
  `{"schema_version":"soulsbench/v1","episode_id":…,"deadline_ms":200,"action_labels":[16 labels],"state":{CombatState}}`.
  It must answer `200 {"discrete_id": 0..15}` (extra keys are ignored) within the deadline.
* `POST {base}/reset` with `{"episode_id": …}`. This is optional; errors are ignored.

**OpenAI Decisions** (`--adapter openai`, public beta since 2026-10) is
`POST https://api.openai.com/v1/decisions` with model `gpt-6-luna`
([guide](https://developers.openai.com/api/docs/guides/decisions)). It is the same kind of model as
Jev, so [`openai_decisions.py`](soulsbench/adapters/openai_decisions.py) reuses the System One
adapter (same prompts, state projection and offered actions) and only reshapes the request:
`input` is the state as compact JSON text, and the action is one `choice` question whose
`choices` are `{value: label, description: criterion}`. The key comes from `OPENAI_API_KEY` or
`./openai_api_key.txt` (gitignored). It also takes `@r5` / `@base`.

## Scenarios and reset semantics

* `gundyr_p1`: Iudex Gundyr phase 1, fixed SoulsGym start pose, 180 s limit. Gundyr's in-game HP is
  refilled every tick, so he never transforms into phase 2. His damage is tracked against his 1037 HP
  pool; this is SoulsGym's own phase-1 convention. The **player's HP is real**, so estus heals. The
  game's death is disabled (SoulsGym's no-death flag), so HP ≤ 1 counts as death and ends the episode.
* `gundyr_p1_randstart`: same, with a start pose drawn from SoulsGym's 11 poses by seed.
* `gundyr_full`: the whole fight. Gundyr's HP is real: he transforms into phase 2 at about half HP
  (`target.boss_phase` switches to 2), and a win means he dies on screen. The bridge lifts the
  no-death flag at that moment so the kill animation plays. Limit 300 s. An episode that reached
  phase 2 or killed him forces a game reload before the next one (~20–30 s), because only a reload
  restores phase 1. Phase-2 versions of attacks have their own hit windows (`p2:<anim>` in the
  calibration file), and his transformation (`Attack1500`) is reported as its own situation.

**Reset is SoulsGym's arena reset, not a save state; DS3 has none.** It teleports the player and
Gundyr to their start poses, refills HP and stamina, resets animations and re-acquires lock-on. When
estus was spent, the bridge first reloads at the Cemetery of Ash bonfire, which refills the flask in
the game itself. SoulsGym also reloads every 15 minutes, because its input handling degrades over
long sessions. Each reload adds ~15–30 s of wall time.

The character uses SoulsGym's forced Knight SL9 stats and starting gear (Long Sword, Knight Shield).

## Known limitations

* **Estus, poise, i-frame and parry fields are read through pointer chains** taken from The Grand
  Archives cheat table bundled with SoulsGym, for 1.15.2. They are read-only and fall back to `null`
  if a read fails. Verify them with `python -m soulsbench.windows_bridge probe` (Windows, player
  in-game). DS3's inventory count for the Estus Flask never shows the last charge being used: it
  stays at 1 while the HUD shows an empty flask. So the bridge counts each drinking animation itself
  (heal or count drop) and reports `min(start - drinks, memory)`. That was verified against the HUD
  on the first live run.
* **Poise is `null`.** The cheat table's super-armor chain read 0.0/0.0 on every live tick for
  the Knight, so it is not the player's poise; the bridge reports unknown rather than a fake zero.
  The i-frame flag, by contrast, is verified live: it turns on during knockdown animations, exactly
  when DS3 makes a downed player invulnerable.
* Input windows exist only for animations SoulsGym measured; others use the memory-length fallback
  described above. Unknown animation names are logged in `steps.jsonl` (`player_anim`).
* One game, one keyboard: the benchmark cannot run while you use the PC, and it cannot run episodes
  in parallel.

## Development

```bash
uv run pytest            # 100 tests: schema, state, harness, metrics, adapters, bench, HTTP bridge
```

`tests/fakes.py` is a tiny deterministic environment. It is a unit-test double only; there is no
offline backend in the CLI. `tests/test_live_env_logic.py` runs SoulsGym's real `step()` code
through SoulsBench's subclass against a mocked game. To enable it in the WSL venv, install SoulsGym's
pure-Python parts:

```bash
uv pip install --no-deps "soulsgym @ https://github.com/amacati/SoulsGym/archive/29c94d89ac8a47846c619019296d0ba55c6e80d5.zip"
uv pip install "gymnasium>=1.0" numpy psutil pyyaml "iced-x86>=1.21" "pymem>=1.10"
```

Repo map: `soulsbench/schema.py` (contract), `harness.py` (loop and SLA), `metrics.py` (score card),
`bench.py` (interleaved resumable benchmark), `pricing.py`, `adapters/`, `scenarios.py`,
`calibrate.py`, `backends/souls_gym.py` (WSL client), `windows_bridge/` (Windows side: `live_env.py`
SoulsGym subclass and realtime `/act`, `state.py` CombatState builder, `memory.py` extra reads,
`recorder.py` OBS, `server.py` HTTP, `game.py` launch/backup/network), `windows/` (setup, firewall
and run scripts), `scripts/` (local model servers, the full-benchmark driver, chart and clips),
`docs/` (results).

Known operational quirks, all handled but worth knowing: injecting the speedhack into a DS3 that is
already past startup crashes it, so the bridge refuses and asks you to let it launch the game;
DS3 occasionally hangs on its own (the bench stops and resumes); Kev's server ignores SIGTERM.
