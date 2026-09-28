"""Repo-local declarations for tests/test_openapi_contract.py (atrium-project#32 round 2).

Never vendored, never in para-drift, never in the ruff [format] exclude — unlike the
canonical test that reads it, this file's content is per repo by design: which services the
repo runs, where their committed specs live, which settings could reach a spec, and which
requirement files pin fastapi and pydantic. See the canonical test's docstring.
"""

from __future__ import annotations

#: One entry per HTTP service of this repo. `primary`: the domain endpoints whose JSON 200
#: must be a named model (strategy §4.2).
SERVICES = [
    {
        "service": "atrium-alto-postprocess",
        "app": "service.text_api:app",
        "spec": "service/openapi.json",
        "primary": ["/process"],
    },
]

#: Settings besides every [limit] variable (which the test perturbs from tool_limits.LIMITS)
#: that a deployment changes and that must not change the spec: the CORS origins, where the
#: models come from, and the batch record directory document_hook reads.
ENV_PERTURB = {
    "ALLOWED_ORIGINS": "https://example.org",
    "MODEL_DIR": "/nonexistent/models",
    "GPT2_MODEL_NAME": "perturbed/model",
    "LAYOUT_MODEL_PATH": "perturbed/layoutreader",
    "DOCUMENT_JSON_DIR": "/nonexistent/records",
}

#: Every requirements file a lane or an image installs fastapi or pydantic from: the api image
#: and setup/setup_api_server.sh (service/requirements.txt) and the light and docker-tool test
#: lanes (setup/requirements-test.txt, which the root requirements-test.txt forwards to).
PIN_FILES = ["service/requirements.txt", "setup/requirements-test.txt"]

#: Run before the app is imported (``MODULE:FUNCTION``), or None: the service imports without
#: torch (the models load in the lifespan, which the test never enters).
PREPARE = None
