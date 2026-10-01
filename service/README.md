# ATRIUM Text Processor API Service 🚀

### Goal: Serve historical text cleaning and quality estimation models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the ATRIUM Text Processing pipeline.
It allows users to upload ALTO XML or raw text files to perform intelligent layout analysis,
split-word reconstruction, and line-level quality classification (e.g., `Clear`, `Noisy`, `Trash`,
`Non-text`, `Empty`) using **LayoutLMv3**, **FastText**, and **Qwen2.5-0.5B** [^9] [^2] [^6].
Two frontend variants are included: a **standalone** interface (`frontend/`) and a
**LINDAT-integrated** interface (`frontend-lindat/`).

### Table of contents 📑

* [Service Description 📇](#service-description-)
* [Directory Structure 📂](#directory-structure-)
* [Supported Models 🧠](#supported-models-)
* [Quality Categories 🪧](#quality-categories-)
* [API Usage 📡](#api-usage-)
* [Installation & Setup 🛠](#installation--setup-)
  * [Prerequisites](#1-prerequisites)
  * [Install Dependencies](#2-install-dependencies)
  * [Model Weights](#3-model-weights)
* [Quick API Test Launch 🚀](#quick-api-test-launch-)
* [Launch Instructions](#launch-instructions)
  * [Running the Server 🚀](#running-the-server-)
  * [Standalone Frontend 🖥️](#standalone-frontend-)
  * [LINDAT-integrated Frontend 🎨](#lindat-integrated-frontend-)
* [Configuration (environment) ⚙️](#configuration-environment-)
* [Limits 📏](#limits)
* [Errors 🚨](#errors)
* [OpenAPI (the typed contract) 📜](#openapi-the-typed-contract)
* [Contacts 📧](#contacts-)
* [Acknowledgements 🙏](#acknowledgements-)

---
## Service Description 📇

The API is built using **FastAPI** and is designed to turn raw OCR output into clean, classified text data.
It acts as a bridge between complex NLP models and downstream applications or web interfaces.

Key features:

* **Layout Analysis:** Uses **LayoutLMv3** to correctly reorder tokens from ALTO XML files based on 2D spatial layout, handling multi-column pages [^9].
* **Text Cleaning:** Automatically detects and merges hyphenated words split across lines using ALTO `SUBS_TYPE` / `SUBS_CONTENT` attributes and regex-based reconstruction.
* **Quality Classification:** Classifies every line with a composite **quality score** built from structural detectors (strange symbols, mid-word uppercase, letter–digit–letter fusions, gibberish, fused/rotated tokens) and **Qwen2.5-0.5B** perplexity, implemented in `text_util.py` [^6]. The category is then assigned from quality-score thresholds plus named overrides.
* **GPU Support:** Automatically detects and utilises CUDA devices for inference if available [^3].
* **Two Frontend Variants:** A self-contained standalone interface for direct use, and a LINDAT-integrated interface for deployment within the LINDAT Common framework.
* **CORS Support:** Cross-Origin Resource Sharing is configurable via the `ALLOWED_ORIGINS` environment variable. The code default is `*` (every origin); `docker-compose.yml` supplies a narrower `http://localhost:8080,http://localhost:5500` default of its own, which applies only under compose.

## Directory Structure 📂

The service logic resides in the `service/` directory, while models are expected in a `models/` directory at the project root.

```text
atrium-alto-postprocess/
├── v3/                          # 📦 LayoutReader helper scripts
├── models/                      # 📦 Model weights (downloaded externally)
│   └── lid.176.bin              # FastText language identification binary
├── service/                     # 🚀 API source code
│   ├── text_api.py              # FastAPI application entry point
│   ├── text_inference.py        # Model manager (LayoutLMv3, FastText, Qwen2.5-0.5B)
│   ├── utils.py                 # XML parsing, box normalisation, cleaning logic
│   ├── frontend/                # 🖥️  Standalone frontend (no external dependencies)
│   │   ├── index.html           # Self-contained web interface
│   │   └── script.js            # Vanilla JS — no jQuery, no build step required
│   ├── frontend-lindat/         # 🎨 LINDAT-integrated frontend
│   │   ├── index.html           # Interface styled for lindat-common
│   │   └── script.js            # JS adapted to the lindat-common webpack bundle
│   ├── requirements.txt         # Python dependencies
│   └── README.md                # API service documentation
├── setup/                       # ⚙️ Configuration and setup files
│   └── setup_api_server.sh      # Sets up virtual environment and installs dependencies
├── text_util.py          # Structural quality detectors and categorisation logic
├── README.md                    # Project overview and documentation (this file)
├── LICENSE
└── ...                          # Other project files (scripts, data samples, paradata)
```


## Supported Models 🧠

The pipeline applies three models in sequence, balancing structural layout understanding with semantic quality estimation.

| Model            | Purpose                                                                                                 | Source             |
|------------------|---------------------------------------------------------------------------------------------------------|--------------------|
| **LayoutLMv3**   | **Reading Order:** Reorders tokens in ALTO XML files based on 2D bounding-box layout.                   | by `hantian` [^9]  |
| **FastText**     | **Language ID:** Identifies the language of each line as a pre-filter signal.                           | by `facebook` [^2] |
| **Qwen2.5-0.5B** | **Perplexity:** Measures how linguistically "surprising" a line is — elevated scores suggest OCR noise. | by `Qwen` [^6]     |

> [!NOTE]
> The category is decided by the composite **quality score** — a weighted sum of nine structural, language and
> perplexity signals routed through thresholds, with named overrides — not by a fixed detector decision-tree.
> Perplexity is one weighted signal; on short 1–2 word lines it is capped before scoring because the LM has too
> little context. `distilgpt2` remains available as an English-only alternative via the `GPT2_MODEL_NAME`
> environment variable (re-tune `PERPLEXITY_THRESHOLD_MAX`, see [Troubleshooting](#hardware--configuration-troubleshooting)).
> Full logic: main [README → Composite Quality Score](../README.md#composite-quality-score) and
> [Categorisation Logic](../README.md#categorisation-logic).

## Quality Categories 🪧

The service classifies every text line into one of five categories. The first two (`Empty`, `Non-text`)
are assigned by a fast CPU pre-filter before any model inference. The remaining three are assigned by
`text_util.categorize_line()` from the composite **quality score**, after immediate overrides.

| Label         | Description                                                           | Primary Signal                                                                                    |
|---------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `Clear` 🟢    | **High quality.** Ready for downstream NLP.                           | `quality_score ≥ CATEG_NOISY_SCORE_MAX` (0.80), or a low-perplexity override.                     |
| `Noisy` 🟡    | **Usable but degraded.** Minor OCR artefacts, recoverable downstream. | `CATEG_TRASH_SCORE_MAX` (0.55) ≤ `quality_score` < `CATEG_NOISY_SCORE_MAX` (0.80).                |
| `Trash` 🔴    | **Structurally corrupt.** Re-process with another OCR / HTR tool.     | `quality_score < CATEG_TRASH_SCORE_MAX` (0.55), or a hard override (all-caps/no-vowel, inverted). |
| `Non-text` 🔵 | **No meaningful text.** Purely numeric / separator content.           | CPU pre-filter: dates, page numbers, archive/stamp codes, or digit ratio > 40 % on short lines.   |
| `Empty` ⚪     | **Blank line.** Whitespace only.                                      | `word_count == 0` / whitespace only.                                                              |

> [!NOTE]
> The thresholds and the full set of overrides (hard-sweep, inverted-scan, low-perplexity-clear,
> mostly-readable cap, and the document/page post-passes) are documented once in the main
> [README → Categorisation Logic](../README.md#categorisation-logic) and are not duplicated here.

> [!IMPORTANT]
> `/process` classifies through **the same scoring function as the batch pipeline**
> (`classify_TEXT.score_line()`), so the API and a pipeline run return the same category for the
> same line. This was not always true: the endpoint used to assemble its own signals and had
> drifted — it skipped the language remap, the two-tier trust scaling and `SHORT_PPL_CAP`, and it
> never passed `orig_lang_score`, leaving that argument at its `1.0` default. Three Trash routes
> that key on low language confidence (`rule_hard_sweep`, `rule_extreme_ppl`, `rule_wqx_rot`) could
> therefore never fire from the API, which returned `Noisy` for lines the pipeline calls `Trash`.
>
> Two consequences worth knowing when comparing API output against a batch CSV:
> * the service has no separate pre-repair text, so `garbage_density` and `vowel_ratio` are computed
>   on the submitted line, whereas the pipeline computes them on the original pre-repair line;
> * document- and page-level smoothing (`pp_*`) is a batch pass over a whole document and does not
>   apply to single-line API calls.


## API Usage 📡

### Endpoints 🔗

| Method | Path       | Description                                                                                                                                                                                            |
|--------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GET`  | `/`        | Serves the standalone `index.html` interface for manual testing.                                                                                                                                       |
| `GET`  | `/info`    | Service identity + capabilities: `service`, `version`, `endpoints`, `limits` (every [limit](#limits)), `limits_meta` (the variable behind each), plus status, device, line fields, quality categories. |
| `GET`  | `/health`  | Liveness probe — 200 always, even mid-shutdown. `?deep=true` also checks the quality/language models are loaded (503 on failure or while draining).                                                    |
| `GET`  | `/ready`   | Readiness probe (issue #55) — 503 until model load finishes, 200 while serving, 503 the instant `SIGTERM` arrives. The Kubernetes `readinessProbe`/`startupProbe` target.                              |
| `POST` | `/process` | Uploads a file for layout analysis, cleaning, and line-level classification.                                                                                                                           |

### Request Example 💻

**Endpoint:** `/process`

**Parameters (Form Data):**

* `file`: The document file (`.xml` ALTO, `.txt` plain text, or `.json` generic OCR JSON) — or (#31) any other
  text-bearing document: PDF, DOCX, ODT/ODS/ODP, XLSX, PPTX, EPUB, RTF, HTML/hOCR, PAGE XML, TEI, Markdown,
  CSV/TSV, JSON Lines.
* `task_type`: `alto`, `text`, `json`, `document`, or `auto` (default). `auto` maps `.txt`→`text` and
  `.json`→`json` by extension; `.xml` and every other upload are decided **from the bytes** — an ALTO root →
  `alto`, anything else readable → `document`. Any other value is a `422` (atrium-project#32 round 2; it used
  to be read as `text`). An unsupported file (image, legacy `.doc`, other binary) is a `415`
  `unsupported_media_type` naming the reader's code (a `400` before round 2); a document over a
  [limit](#limits) is a `413` (`422` for the reader's time limit) with `reason: "limit_exceeded"`; an
  unreadable document (encrypted, corrupt, no text) or a JSON upload that does not parse is a `422`. See
  [Errors](#errors).
* `document_json` (optional): a baseline ATRIUM document record, or an AMČR seed (`doc_id`, `source`). The
  response then carries `document_json`: the record with this tool's `pages`/`lines` fields merged in, and
  `source` given what it lacks. This service reads the source, so it records `source.origin`; every value
  the record already has is kept (first writer wins), and the upload's `sha256` never joins a seed's
  `sha512`, since the ALTO is not the archive's original (atrium-project#71). A
  record that cannot be opened (not JSON, not an object, a newer `schema_version` major) is a `422`
  `invalid_record`; an empty part counts as none.
* `document_record` (optional, **deprecated**): the earlier name of `document_json`; its record comes back as
  `document_json_out`. Both names keep working; send one of them, not both (`422`).

Every response carries `limits_applied`: the limits that shaped the result without refusing it (see
[Limits](#limits)), `[]` when none did.

Every response also carries **`paradata`**, the call's provenance (atrium-project#71): one Process Run
Crate `CreateAction`, built by `atrium_rocrate.create_action()`. Its `@id` is the call's `run_uuid`,
which also stamps every block the call wrote into the record; `object` is the upload (by content hash)
and the record sent, `result` the blocks written and the classified lines (`cleaned_lines.json`);
`agent` is `ATRIUM_RUN_AGENT` when set. The service writes no paradata file. An error response carries
no action. Hub
[`docs/rocrate_export.md`](https://github.com/ufal/atrium-project/blob/main/docs/rocrate_export.md) §5
describes it.

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/page_01.xml" \
  -F "task_type=auto"
```

For a generic JSON OCR-engine export, the endpoint reads the lines exactly as the batch
pipeline's `extract_JSON_2_TXT.py` reads one page (`page_text_lines`): the same key whitelist
(`content`, `text`, `line`, `word`, …), the page object of a split page, and each text once at
line granularity — a line's words and a page's whole-text field are not repeated (#31 Phase 5):

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/page_01.json" \
  -F "task_type=auto"
```

### Response Schema

The top-level `type` field is `alto_xml`, `plain_text`, `json`, or `document`, matching the routed `task_type`.
A `document` result also carries `format` (the detected kind), `origin` (the truthful `source.origin` class),
a `pages` list (`page`, `page_label`, `lines`, PDF `text_layer` / `needs_ocr_reason`), and `page` /
`page_label` on every line, whose `line_num` restarts per page as in the batch `DOC_LINE_CATEG`. Lines are read
and shaped by the same `text_formats.py` code as the batch text-lines method (blank lines dropped, lines over
1000 characters wrapped). With a record (`document_json`), lines accrete per page — except for born-digital
uploads (DOCX, visible-text PDF, …), whose record belongs to llm-enrich's `digital-convert` (atrium_document
§1a) and comes back as it was sent. Each item in `cleaned_lines` carries the fields used by the
classification pipeline. Every field is typed in [`openapi.json`](openapi.json) (`ProcessResponse`,
`AltoLine`, `AltoPage`); the table below is the short form.

```json
{
  "type": "alto_xml",
  "filename": "page_01.xml",
  "cleaned_lines": [
    {
      "line_num": 1,
      "text": "The quick brown fox jumps over the lazy dog.",
      "lang": "eng",
      "lang_score": 0.9821,
      "perplexity": 12.5,
      "sym_count": 0,
      "upper_count": 0,
      "word_weird": 0.0,
      "quality_score": 0.9501,
      "category": "Clear"
    },
    {
      "line_num": 2,
      "text": "TYRSOVA5===aras T>r«l",
      "lang": "ces",
      "lang_score": 0.4201,
      "perplexity": 4800.0,
      "sym_count": 2,
      "upper_count": 0,
      "word_weird": 0.85,
      "quality_score": 0.1205,
      "category": "Trash"
    },
    {
      "line_num": 3,
      "text": "1956–1959",
      "lang": "N/A",
      "lang_score": 0.0,
      "perplexity": 0.0,
      "sym_count": 0,
      "upper_count": 0,
      "word_weird": 0.0,
      "quality_score": 0.0,
      "category": "Non-text"
    }
  ]
}
```

**Response fields:**

| Field           | Type   | Description                                                                                                                                                                                                                |
|-----------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `line_num`      | int    | 1-based line position after layout reordering.                                                                                                                                                                             |
| `text`          | string | Cleaned line text with split-word merges applied.                                                                                                                                                                          |
| `lang`          | string | ISO language code predicted by FastText (e.g., `eng`, `ces`).                                                                                                                                                              |
| `lang_score`    | float  | FastText confidence score `[0, 1]`.                                                                                                                                                                                        |
| `perplexity`    | float  | Qwen2.5-0.5B perplexity. `0` means the line was pre-filtered and inference was skipped.                                                                                                                                    |
| `sym_count`     | int    | Tokens containing characters outside the allowed internal set (`detect_strange_symbols`).                                                                                                                                  |
| `upper_count`   | int    | Tokens with mid-word uppercase artefacts — Patterns 1–3 (`detect_mid_uppercase`).                                                                                                                                          |
| `word_weird`    | float  | Mean per-word weirdness score `[0, 1]`; combines strange-symbol, repeated-char, LDL-fusion, mid-uppercase and mirror-OCR (`w` / caps-prefix) signals; `0` = fully clean.                                                   |
| `quality_score` | float  | Composite quality score `[0, 1]`; weighted sum of nine signals (valid-word ratio, word-weirdness, perplexity, length, garbage density, vowel quality, language confidence, gibberish, fused-word ratio); higher = cleaner. |
| `category`      | string | One of: `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty`.                                                                                                                                                                    |


## Installation & Setup 🛠

### 1. Prerequisites

* **Python 3.11** virtual environment [^5] (matches `python:3.11-slim`, the image base, and the CI lane — atrium-project#64).
* **Standard CPU** (sufficient for inference; GPU recommended for batch processing).
* **CUDA-capable GPU** (optional — auto-detected at startup for faster inference) [^3].
* **NodeJS** (only required for the **LINDAT-integrated** frontend — `export NODE_OPTIONS=--openssl-legacy-provider` is a common fix for Webpack 4 compatibility with NodeJS 17+).

### 2. Install Dependencies

Clone the repository and run the setup script from the project root. It creates a virtual
environment, installs all Python dependencies, fetches the `v3/` LayoutReader scripts via
sparse checkout, and downloads the FastText binary:

```bash
git clone [https://github.com/ufal/atrium-alto-postprocess.git](https://github.com/ufal/atrium-alto-postprocess.git)
cd atrium-alto-postprocess
chmod +x setup/setup_api_server.sh
./setup/setup_api_server.sh
```

Key libraries: `fastapi`, `uvicorn`, `python-multipart`, `torch`, `transformers`, `fasttext`, `lxml`, `numpy`.
Full list in `service/requirements.txt` for manual installation if needed.

> [!NOTE]
> The virtual environment name is set in `setup/setup_api_server.sh` and can be changed to match an existing environment.

### 3. Model Weights

The setup script downloads the FastText binary automatically. If you prefer to download it manually:

```bash
mkdir -p models
wget "[https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin](https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin)" \
     -O models/lid.176.bin
```

> [!NOTE]
> LayoutLMv3 and Qwen2.5-0.5B are downloaded and cached automatically by Hugging Face Transformers on the first run [^9] [^6].

## Quick API Test Launch 🚀

```bash
source venv/bin/activate
python service/text_api.py
```

The server starts at `http://0.0.0.0:8000`. The standalone frontend is served at `/`.
Send a test request in a second terminal:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@data_samples/ALTO/CTX195603828.alto.xml" \
  -F "task_type=alto"
```

## Launch Instructions

### Running the Server 🚀

Activate your virtual environment and start the API with hot-reloading (useful during development):

```bash
cd atrium-alto-postprocess
source venv/bin/activate          # or: source venv-api/bin/activate
uvicorn service.text_api:app --reload
```

The server will be available at `http://0.0.0.0:8000`.

---

### Standalone Frontend 🖥️

`service/frontend/` is a self-contained interface with no build step or external framework required.
It is served directly by the FastAPI server at `http://localhost:8000` and works out of the box.

**To use it**, simply start the server (see above) and open `http://localhost:8000` in your browser.

Features:
- Drag-and-drop or click-to-upload for `.xml`, `.txt` and `.json` files and (#31) PDF, DOCX, ODT, XLSX, PPTX, EPUB,
  RTF, HTML/hOCR, Markdown and CSV documents.
- Processing mode selector (`auto` / `alto` / `text` / `json` / `document`).
- Results table with `Sym`, `Upper`, and `PPL` columns aligned to `text_util.py`.
- Category breakdown bar showing counts for all five labels.
- Raw extracted text toggle.

> [!NOTE]
> If you are running the frontend from a local dev server (e.g. Live Server on port `5500`),
> `script.js` automatically redirects API calls to `http://localhost:8000`.

---

### LINDAT-integrated Frontend 🎨

`service/frontend-lindat/` is the frontend variant styled and bundled for deployment within the
[LINDAT Common](https://github.com/ufal/lindat-common) framework. It requires NodeJS and the
`lindat-common` webpack build.

Open a **second terminal window** alongside your running server and follow these steps:

**1. Place the project inside `lindat-common`:**

```bash
git clone [https://github.com/ufal/lindat-common.git](https://github.com/ufal/lindat-common.git)
cd lindat-common
cp -r /path/to/atrium-alto-postprocess .
```

**2. Install NodeJS and dependencies:**

```bash
curl -o- [https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh](https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh) | bash
nvm install stable
nvm use stable
export NODE_OPTIONS=--openssl-legacy-provider
npm install
```

**3. Start the webpack dev server:**

```bash
make run
```

Expected output:

```
> lindat-common@3.5.0 start
> webpack-dev-server -p --debug --quiet

> Project is running at http://localhost:8080/
> webpack output is served from /
> Content not from webpack is served from /home/.../lindat-common
```

Open `http://localhost:8080` and navigate to the
`atrium-alto-postprocess/service/frontend-lindat` directory in the file tree.

For further details on the LINDAT development workflow see the
[LINDAT Common Development Guide](https://github.com/ufal/lindat-common/?tab=readme-ov-file#development).


---

## Hardware & Configuration Troubleshooting

* **GLM-4v VRAM Requirements:** The GLM-4v Vision-Language Model requires massive GPU memory. You **must have a GPU
with at least 48 GB of VRAM** (e.g., an NVIDIA RTX A6000 or a multi-GPU setup) to run the extraction pipeline
successfully. Running this on consumer GPUs (like a 3090/4090) will likely result in Out-Of-Memory (OOM) crashes.
* **Perplexity Threshold Coupling:** The service uses **Qwen2.5-0.5B** by default, matched to `PERPLEXITY_THRESHOLD_MAX
= 1000.0` in `config.txt`. If you switch the perplexity model via the `GPT2_MODEL_NAME` environment variable
(e.g., to the English-only `distilgpt2`), you **must** recalibrate `PERPLEXITY_THRESHOLD_MAX` — perplexity scales differ
wildly between architectures (≈ `3000.0` suits `distilgpt2`), so a value tuned for one model is mis-calibrated for the other.

---

## Configuration (environment) ⚙️

| Variable              | Default   | Meaning                                                                       |
|-----------------------|-----------|-------------------------------------------------------------------------------|
| `PORT`                | `8000`    | port the service **binds**, and the one `service/healthcheck.py` probes       |
| `HOST`                | `0.0.0.0` | bind address. ⚠️ see the warning below                                        |
| `GRACEFUL_SHUTDOWN_S` | `20`      | seconds uvicorn waits for in-flight requests before closing them              |
| `RELOAD`              | `false`   | filesystem auto-reload — development only, never in a deployment              |
| `LOG_LEVEL`           | `INFO`    | root logger level for the `python service/text_api.py` start path (issue #61) |
| `ALLOWED_ORIGINS`     | `*`       | CSV of CORS origins — no shared default across the five services              |
| `MAX_UPLOAD_MB`       | `25`      | canonical upload limit — no shared default across the five services           |
| `MODEL_DIR`           | see below | directory models are loaded from                                              |
| `GPT2_MODEL_NAME`     | see below | quality-estimation model id                                                   |

This table is the deployment-facing subset. The complete ledger — every variable this
image reads, including `MODEL_DIR`/`GPT2_MODEL_NAME`/`LAYOUT_MODEL_PATH`/`LANGID_CONFIG`
and the other algorithmic knobs a deployment does not normally touch — is
[`.env.example`](../.env.example) at the repo root, whose layout is fixed by
`docs/templates/env.example.template` in ufal/atrium-project. The cross-service operator
reference is `docs/k8s_deployment.md` in that same repo.

`PORT` and `HOST` are read by `service/text_api.py`'s `__main__` block, which is what the `api` image's `ENTRYPOINT` runs.

alto-postprocess is the reference implementation for this contract: it has honoured
`PORT`/`HOST` since issue #55, and issue #58 brought the other four services into line with
it. Its entrypoint is `python service/text_api.py` (a script launch, made viable by the
`sys.path` bootstrap at the top of that file) rather than the `python -m service.api` the
other four use; the environment contract is identical either way.

> ⚠️ `HOST=127.0.0.1` yields a container that reports **healthy** and serves nobody:
> `service/healthcheck.py` always probes loopback by design and never reads `HOST`, so a
> loopback bind passes every probe while being unreachable from outside the container.

## Limits

Every limit this service has (atrium-project#53). Each is an environment setting, declared once in
[`tool_limits.py`](../tool_limits.py), reported with its current value in `GET /info` `limits` and with the
variable that sets it in `limits_meta`. The `[TEXT_INGEST]` caps are also keys of the config file
(`LANGID_CONFIG`, `setup/config.txt`), which the batch pipeline reads: the variable wins over the file, and the
file over the default below. A malformed value stops the service at startup, naming it. Over a limit the
service **refuses** (`reason: "limit_exceeded"`) or **processes the input in full** and says how the limit
shaped the result, in the response's `limits_applied` (and the paradata). `tests/test_limits_contract.py`
checks this table against `tool_limits.py` and `.env.example`.

| Key (`/info`)        | Variable                                | Default | Unit    | Over the limit                                                                                              |
|----------------------|-----------------------------------------|---------|---------|-------------------------------------------------------------------------------------------------------------|
| `max_upload_mb`      | `MAX_UPLOAD_MB`                         | 25      | MB      | 413 `limit_exceeded` — per part: the file and the document record                                           |
| `max_file_mb`        | `ATRIUM_TEXT_INGEST_MAX_FILE_MB`        | 256     | MB      | 413 `limit_exceeded` (`too_large`); a ZIP member over it is not read — `skipped` note                       |
| `zip_max_members`    | `ATRIUM_TEXT_INGEST_ZIP_MAX_MEMBERS`    | 10000   | members | 413 `limit_exceeded` (`zip_limits_exceeded`)                                                                |
| `zip_max_total_mb`   | `ATRIUM_TEXT_INGEST_ZIP_MAX_TOTAL_MB`   | 1024    | MB      | 413 `limit_exceeded`                                                                                        |
| `zip_max_member_mb`  | `ATRIUM_TEXT_INGEST_ZIP_MAX_MEMBER_MB`  | 256     | MB      | 413 `limit_exceeded`                                                                                        |
| `zip_max_ratio`      | `ATRIUM_TEXT_INGEST_ZIP_MAX_RATIO`      | 200     | ratio   | 413 `limit_exceeded`                                                                                        |
| `max_pages`          | `ATRIUM_TEXT_INGEST_MAX_PAGES`          | 20000   | pages   | 413 `limit_exceeded` (`too_large`)                                                                          |
| `max_lines_per_page` | `ATRIUM_TEXT_INGEST_MAX_LINES_PER_PAGE` | 100000  | lines   | a paged format: 413 `limit_exceeded`; any other: split onto continuation pages, read in full — `split` note |
| `reader_timeout_s`   | `ATRIUM_TEXT_INGEST_READER_TIMEOUT_S`   | 300     | s       | 422 `limit_exceeded` (`timeout`, PDF reader) — a per-input budget, a retry will not help                    |
| `max_line_chars`     | `ATRIUM_TEXT_INGEST_MAX_LINE_CHARS`     | 1000    | chars   | wrapped at a word boundary, classified in full as several lines — `split` note                              |
| `odf_repeat_cap`     | `ATRIUM_TEXT_INGEST_ODF_REPEAT_CAP`     | 100     | repeats | repeated ODS/ODT cells or rows past it are not read — `trimmed` note                                        |
| `pdf_object_cap`     | `ATRIUM_TEXT_INGEST_PDF_OBJECT_CAP`     | 20000   | objects | the text layer is judged on the first N objects (the text is read in full) — `sampled` note                 |
| `lr_chunk_size`      | `LR_CHUNK_SIZE`                         | 350     | lines   | an ALTO page is ordered window by window — `split` note                                                     |
| `lr_min_chunk_size`  | `LR_MIN_CHUNK_SIZE`                     | 50      | lines   | out of memory below it: the page keeps its document order — `skipped` note                                  |
| `ppl_batch_lines`    | `PPL_BATCH_LINES`                       | 128     | lines   | lines per perplexity pass; a failed batch scores its lines 99999 — `skipped` note                           |
| `ppl_max_tokens`     | — (derived from `GPT2_MODEL_NAME`)      | —       | tokens  | the perplexity model reads the first N tokens of a line — `trimmed` note (`null` until the model loads)     |

Platform limits (not settings): libxml2's default limits (parsed without `huge_tree`); Starlette's multipart
defaults.

## Errors

Every error has one JSON body (hub `docs/agent_skill_strategy.md` §4.4, atrium-project#32 item 2):
`{"status": <int>, "reason": <code or null>, "detail": "<text>"}`. `detail` starts with alto's own reader code
where there is one (`too_large: …`, `corrupt: …`); a `limit_exceeded` body adds `limit` (`key`, `env`, `value`,
`observed`, `unit`), and a request-validation 422 adds `errors`.

Since atrium-project#32 round 2 a reader refusal also carries the reader's code as `cause` (informational,
not registered: it may change; `reason` is the stable one), and a `415` lists the extensions the readers take
as `accepted`.

| Code | `reason`                 | When                                                                                                                                                                           |
|------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 413  | `limit_exceeded`         | over `MAX_UPLOAD_MB` or a `[TEXT_INGEST]` size/count limit (`too_large`, `zip_limits_exceeded`)                                                                                |
| 415  | `unsupported_media_type` | a kind of file this service does not read (`binary_content`, `image_needs_ocr`, `legacy_office_unsupported`, `archive_unsupported`); a 400 before round 2                      |
| 422  | `invalid_record`         | the `document_json` / `document_record` part cannot be opened (not UTF-8 JSON, not an object, a newer `schema_version` major)                                                  |
| 422  | `limit_exceeded`         | the PDF reader's time limit (`timeout`)                                                                                                                                        |
| 422  | `null`                   | a supported file that cannot be read (`corrupt`, `encrypted`, `malformed`, `no_text`, …), a JSON upload that does not parse, or request validation (an unknown `task_type`, …) |
| 500  | `null`                   | processing failure                                                                                                                                                             |
| 501  | `null`                   | this deployment lacks a reader's optional dependency (`dependency_missing`: pypdfium2, lxml); a 400 before round 2                                                             |
| 503  | `null`                   | the replica is shutting down — retry against a live one                                                                                                                        |

```json
{"status": 413, "reason": "limit_exceeded",
 "detail": "too_large: 25000 pages > MAX_PAGES=20000 (setting: ATRIUM_TEXT_INGEST_MAX_PAGES, or [TEXT_INGEST] MAX_PAGES in the config)",
 "limit": {"key": "max_pages", "env": "ATRIUM_TEXT_INGEST_MAX_PAGES", "value": 20000, "observed": null, "unit": "pages"}}
```

## OpenAPI (the typed contract)

The service's OpenAPI document is committed as [`service/openapi.json`](openapi.json) and
attached to every release as `openapi.json` with its `openapi.json.sha256` (atrium-project#32
round 2). It is what a client is generated from: every request and response field is typed,
every error response is the `ErrorBody` above, the registered `reason` codes are listed in
`x-atrium-reason-codes`, and a returned record is typed by the vendored record schema
(`AtriumDocument`). `GET /info` reports `openapi_sha256`, the digest of the spec the running
image serves — equal to the release's `openapi.json.sha256` for an image built from that tag.
`document_record` and `document_json_out` are marked `deprecated` there.

- **After an API change**, regenerate and commit it:
  `python atrium_openapi.py export --app service.text_api:app --out service/openapi.json`.
  `tests/test_openapi_contract.py` fails while it is stale.
- **Compatibility.** Each release compares its spec with the previous release's
  (`release.yml`, `atrium_openapi.py compare` with oasdiff): a breaking change fails the
  release unless the major version went up (for 0.x, that means 1.0), and a removed reason
  code always fails. New fields, endpoints and reason codes are additive.
- **fastapi and pydantic are pinned** exactly (`service/requirements.txt`,
  `setup/requirements-test.txt`): the spec is generated by them. Bump both by hand and regenerate.
- **Tests.** `tests/test_api_contract.py` drives `/process` with canned model results and holds every
  response — 200s and refusals — to the published schema; `tests/test_openapi_contract.py` (vendored from
  the hub) checks the committed spec itself.

## Shutdown behavior 🛑

Issue [#55](https://github.com/ufal/atrium-project/issues/55). The published `api` image
(`ghcr.io/ufal/atrium-alto-postprocess:<version>-api`, new in that issue — before it this
service was only reachable via a compose entrypoint override, so no API image existed to
deploy) declares `HEALTHCHECK` (shallow `GET /health`, via the vendored
`service/healthcheck.py`) and `STOPSIGNAL SIGTERM`. `service/text_api.py`'s own
`__main__` block — which is this repo's production start path — passes
`timeout_graceful_shutdown` (`GRACEFUL_SHUTDOWN_S`, default 20s). See
[Configuration (environment) ⚙️](#configuration-environment-) for the full set.

On `SIGTERM` the service flips `GET /ready` to **503** at once so an orchestrator stops
routing to it, answers new `/process` calls with 503, and lets in-flight processing finish
before exiting. `GET /health` deliberately stays 200 throughout — a liveness probe failing
mid-shutdown would get the container killed before the drain completed.

Inference now runs in a worker thread (`asyncio.to_thread`) rather than inline on the
event loop. That was a prerequisite, not a tidy-up: uvicorn's `SIGTERM` handler is an
event-loop callback, so while a synchronous `process_alto()` held the loop the signal
could not be processed at all. Draining also matters here specifically because `/process`
writes a `delete=False` temp file that only its own `finally` clause removes — a request
killed by `SIGKILL` mid-flight leaves that file behind.

⚠️ Model load failure at startup deliberately raises, so a misconfigured deployment
crash-loops on the startup probe rather than sitting "not ready" forever. That is intended
— see `docs/k8s_deployment.md` ("Known limits") in the hub.

A clean shutdown exits **143** (128 + SIGTERM), not 0: uvicorn re-raises the captured
signal on purpose so a supervisor sees the real cause. That is a normal stop, not a crash.

## Contacts 📧

**For support write to:** lutsai.k@gmail.com — responsible for this GitHub repository [^8] 🔗

## Acknowledgements 🙏

* **Developed by** UFAL [^7] 👥
* **Funded by** ATRIUM [^4] 💰
* **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
* **Models used:**
  - **LayoutLMv3** for reading-order layout analysis [^9]
  - **FastText** for language identification [^2]
  - **Qwen2.5-0.5B** for perplexity estimation [^6]

**©️ 2026 UFAL & ATRIUM**

---

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://developer.nvidia.com/cuda-python
[^4]: https://atrium-research.eu/
[^5]: https://docs.python.org/3/library/venv.html
[^6]: https://huggingface.co/Qwen/Qwen2.5-0.5B
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^9]: https://github.com/ppaanngggg/layoutreader
