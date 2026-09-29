# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

# --- Provenance (flows into atrium_paradata.py via ENV) ---
ARG ATRIUM_RUNNER_IMAGE=""
ARG ATRIUM_RUNNER_REPO="https://github.com/ufal/atrium-alto-postprocess"
ARG ATRIUM_RUNNER_REF=""
# CPU torch by default (alto-tools method needs no GPU; LayoutReader runs on CPU too,
# just slower). Override to a CUDA wheel index (e.g. .../whl/cu121) to build a GPU
# image for fast LayoutReader runs.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

ENV ATRIUM_RUNNER_IMAGE=${ATRIUM_RUNNER_IMAGE} \
    ATRIUM_RUNNER_REPO=${ATRIUM_RUNNER_REPO} \
    ATRIUM_RUNNER_REF=${ATRIUM_RUNNER_REF} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    MODEL_DIR=/app/models \
    LANGID_CONFIG=/app/setup/config.txt

# (#3) The per-document line CSVs now carry extra diagnostic boolean and string columns
# (including original_text, original_lang, and rule flags). This aggregation
# ignores them — page stats are unchanged — but a future revision could
# emit per-page rule-frequency sums from them.

# ── Distro security patches, applied at build time ───────────────────────────
# `python:3.11-slim` is a floating TAG, and nothing in this ecosystem bumps it:
# no repo declares a `docker` dependabot ecosystem (docker_gha_roadmap.md, H6),
# so the base layer is whatever Docker Hub last rebuilt. On 2026-09-13 that layer
# carried perl-base 5.40.1-6 with three FIXABLE CRITICAL CVEs — CVE-2026-13221,
# CVE-2026-42496 and CVE-2026-8376, all fixed in 5.40.1-6+deb13u1. The release
# gate in atrium-project's docker-tool.reusable.yml ("Fail the release on fixable
# CRITICAL vulnerabilities") blocks on exactly that class, and because the
# promotion step is `if: success()`, a blocked release publishes by DIGEST ONLY —
# the `:<version>` and `:latest` tags are never applied.
#
# It has already cost two releases: translator v1.0.0-beta (2026-09-13, both
# targets) and nlp-enrich v0.20.2 (2026-09-15, run 34970419474, all three
# targets). THIS repo had not been tagged since, which is the only reason it had
# not happened here too — the gate is `if: startsWith(github.ref, 'refs/tags/')`,
# so day-to-day `test` pushes never surface it. (atrium-project#53)
#
# `upgrade` rather than `install --only-upgrade perl-base`, deliberately. The gate
# blocks on *fixable* CRITICALs — precisely those the distro already ships a patch
# for — so the fix that matches the gate's own definition is "apply the distro's
# available patches", not a package name that has to be edited by hand the next
# time a different one is announced.
#
# CACHE INTERACTION, which is what makes this hold rather than run once: the build
# uses `cache-from: type=gha`, so an apt layer high in the file would be served
# from cache forever and silently stop patching. It sits HERE, immediately after
# the ENV block that embeds ATRIUM_RUNNER_REF, because CI passes that as
# `github.ref_name` — a value unique to each release tag. The ENV layer therefore
# changes on every release, busting this layer with it, so every released image is
# scanned against a freshly patched base while day-to-day `test` pushes still hit
# the cache. Do not move this above the ENV block.
#
# One apt layer, not two: the upgrade and the install share a single `apt-get
# update`, so the package lists are fetched once and removed once.
# Guarded by tests/test_dockerfile_security_layer.py (atrium-project#53).
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        build-essential g++ git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) deps first for layer caching. CPU torch pinned, then the unpinned `torch`
#    in the requirements files is already satisfied (stays CPU).
#
#    Runtime requirements only (atrium-project#69, roadmap H4). The image used to install
#    setup/requirements-test.txt (pytest, pytest-cov, httpx, openapi-spec-validator, PyYAML)
#    and setup/requirements-sweep.txt (optuna, SALib, scikit-learn, matplotlib) as well --
#    the sweep file against its own header ("NOT needed by the production pipeline"). Nothing
#    the ENTRYPOINTs reach imports a package only those two carry: the tests run on the CI
#    runner, and the constant sweep (tools/run_optim_pipeline.sh) runs in a host venv and
#    installs its own extras. fastapi/pydantic keep their exact pins in service/requirements.txt.
COPY setup/requirements.txt setup/
COPY service/requirements.txt service/requirements.txt
RUN pip install --index-url ${TORCH_INDEX_URL} torch \
    && pip install -r setup/requirements.txt -r service/requirements.txt

# 2) LayoutReader v3/ (translated from setup_api_server.sh) -> /app/v3 (on sys.path)
#    Required by the GPU extraction method (extract_LytRdr_ALTO_2_TXT.py).
#
#    (#50 rider) Pinned to a commit. `--depth 1` on a branch resolves to whatever
#    the tip is on the day of the build, so two builds a week apart could ship
#    different LayoutReader helpers under the same image tag. A shallow clone
#    cannot take a SHA via `--branch`, hence init + fetch --depth 1 <sha>.
ARG LAYOUTREADER_COMMIT=3b46ee82cd02dcea94ee4064152e5f4de91e205e
RUN git init /tmp/layoutreader \
    && git -C /tmp/layoutreader remote add origin https://github.com/FreeOCR-AI/layoutreader.git \
    && git -C /tmp/layoutreader sparse-checkout init --cone \
    && git -C /tmp/layoutreader sparse-checkout set v3 \
    && git -C /tmp/layoutreader fetch --filter=blob:none --depth 1 origin ${LAYOUTREADER_COMMIT} \
    && git -C /tmp/layoutreader checkout ${LAYOUTREADER_COMMIT} \
    && mv /tmp/layoutreader/v3 /app/v3 \
    && rm -rf /tmp/layoutreader

# 3) Non-root runtime user, created BEFORE the weights (atrium-project#69), so the 1.18 GB
#    file below is written with its final ownership in its own layer. It used to be
#    downloaded as root and chowned by the `chown -R /app` at the end, and overlayfs copies
#    a file up on any metadata change: the image carried the weights twice.
RUN useradd --create-home --uid 10001 atrium

# 4) FastText LID weights -> $MODEL_DIR/lid.176.bin, symlinked to the bare CWD path the
#    batch pipeline loads (classify_TEXT.py:516's FASTTEXT_MODEL default, loaded at :530).
#
#    WHAT THE FILE IS. `facebook/fasttext-language-identification`'s `model.bin`: the NLLB
#    project's fastText language-ID model, 1,176,355,829 bytes, CC BY-NC 4.0 -- NOT the
#    126 MB fastText `lid.176.bin` its local name suggests. The name stays (roadmap H3's
#    rename is retired): it is the default in classify_TEXT.py:516, setup/config.txt:83,
#    service/text_inference.py:83, tools/quality_model/ and tests/test_config_constants.py,
#    and the alto-models volume is seeded from the image only when it is EMPTY, so a renamed
#    file would never reach an existing volume.
#
#    PINNED AND CHECKSUMMED (roadmap H3). The URL names a Hub commit, not `main`, and the
#    file must match the LFS SHA-256 and size. The revision and checksum below are the
#    verified values for facebook/fasttext-language-identification/model.bin. The build
#    still FAILS CLOSED if either value is overridden to empty.
#    To refresh these values from the Hub, inspect the headers of the unpinned URL (no redirect):
#      curl -sI https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin \
#        | grep -iE '^(x-repo-commit|x-linked-etag|x-linked-size):'
#    x-repo-commit -> FASTTEXT_REVISION; x-linked-etag (without the quotes) -> FASTTEXT_SHA256;
#    x-linked-size must equal FASTTEXT_SIZE.
#
#    Hardened fetch, as before:
#      * follows the canonical ?download=true redirect to the LFS CDN,
#      * sends a non-empty User-Agent (Cloudflare rejects some empty-UA bots),
#      * retries transient HTTP errors and connection drops with backoff,
#      * resumes (--continue) the 1.18 GB download instead of restarting it.
#    `-nv` keeps the log readable while still surfacing errors (unlike `-q`). The size
#    check runs before the checksum only so that a truncated body or an HTML error page is
#    named as such; the checksum is what makes the file the pinned one.
ARG FASTTEXT_REVISION=3af127d4124fc58b75666f3594bb5143b9757e78
ARG FASTTEXT_SHA256=8ded5749a2ad79ae9ab7c9190c7c8b97ff20d54ad8b9527ffa50107238fc7f6a
ARG FASTTEXT_SIZE=1176355829
RUN if [ -z "$FASTTEXT_REVISION" ] || [ -z "$FASTTEXT_SHA256" ]; then \
        echo "ERROR: FASTTEXT_REVISION and FASTTEXT_SHA256 are not set, so the fastText weights" >&2; \
        echo "       cannot be fetched pinned. See the comment above this step for the curl" >&2; \
        echo "       one-liner that reads both from the Hugging Face Hub (atrium-project#69, H3)." >&2; \
        exit 1; \
    fi \
    && mkdir -p "$MODEL_DIR" \
    && wget -nv --tries=5 --continue --timeout=60 \
            --retry-connrefused --waitretry=10 \
            --retry-on-http-error=403,408,429,500,502,503,504 \
            --header="User-Agent: atrium-alto-postprocess-docker-build/1.0" \
            "https://huggingface.co/facebook/fasttext-language-identification/resolve/${FASTTEXT_REVISION}/model.bin?download=true" \
            -O "$MODEL_DIR/lid.176.bin" \
    && size="$(stat -c %s "$MODEL_DIR/lid.176.bin")" \
    && if [ "$size" != "$FASTTEXT_SIZE" ]; then \
        echo "ERROR: $MODEL_DIR/lid.176.bin is $size bytes, expected $FASTTEXT_SIZE (truncated or not the model)" >&2; \
        exit 1; \
    fi \
    && echo "${FASTTEXT_SHA256}  $MODEL_DIR/lid.176.bin" | sha256sum -c - \
    && ln -s "$MODEL_DIR/lid.176.bin" /app/lid.176.bin \
    && chown -R atrium:0 "$MODEL_DIR" \
    && chmod -R g=u "$MODEL_DIR"

# 5) source
COPY . .

# 6) ownership of everything else the runtime writes: atrium:0 and group-writable (`g=u`),
#    the arbitrary-UID convention (OpenShift's), atrium-project#69 / roadmap B6.
#    docker-compose.yml runs this image as `user: "${ATRIUM_UID:-10001}:0"`, so on Linux the
#    container can run as the uid that owns the ./data bind mount, and a uid with no passwd
#    entry still reaches /app, /cache, /data and $HOME through group 0. HOME is explicit
#    because without a passwd entry it would be `/`. The default runtime -- uid 10001 as
#    the owner -- is unchanged.
#
#    $MODEL_DIR is PRUNED: it got its ownership in its own layer above, and touching it here
#    would copy the 1.18 GB file into this layer too. The /app/lid.176.bin symlink is
#    changed with `chown -h` and skipped by chmod, which would otherwise follow it to that
#    same file.
RUN mkdir -p /cache/huggingface /data \
    && find /app /cache /data /home/atrium -path "$MODEL_DIR" -prune -o -exec chown -h atrium:0 {} + \
    && find /app /cache /data /home/atrium -path "$MODEL_DIR" -prune -o ! -type l -exec chmod g=u {} +
ENV HOME=/home/atrium
USER atrium

ENTRYPOINT ["python", "run_pipeline.py"]
CMD []


# ---------------------------------------------------------------------------
# API surface — published as :<version>-api (issue #55)
#
# Before this stage existed, alto-postprocess's FastAPI service was reachable only
# through a docker-compose `entrypoint:` override on the BATCH image, so no runnable
# API image was ever published and there was nothing for ARÚP/ARÚB to deploy on
# Kubernetes. service/requirements.txt is already installed in `base` (see the pip
# install above), so this stage only has to declare how to serve.
#
# Entrypoint is `python service/text_api.py` rather than a uvicorn CLI invocation:
# this repo's app deliberately configures logging and reads HOST/PORT/RELOAD in its
# own __main__ block (12-factor VII/XI), and that block is the documented production
# start path. It calls uvicorn.run(..., timeout_graceful_shutdown=...) itself.
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000
# STOPSIGNAL is the default (SIGTERM) — declared explicitly so a future edit cannot
# change it silently; service/text_api.py's lifespan chains to uvicorn's own handler
# for it via serve_lifecycle (service/atrium_service.py).
STOPSIGNAL SIGTERM
ENV PORT=8000 GRACEFUL_SHUTDOWN_S=20
ENTRYPOINT ["python", "service/text_api.py"]
CMD []
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "/app/service/healthcheck.py"]
