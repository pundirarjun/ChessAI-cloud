# Shared Kaggle credential resolution for the persist/upload scripts.
# Source this file, then call kaggle_resolve_credentials.
#
# Resolution order:
#   1. KAGGLE_USERNAME / KAGGLE_KEY already in the environment
#   2. ~/.kaggle/kaggle.json (legacy API credentials file)
#   3. Kaggle notebook secrets (Add-ons -> Secrets: KAGGLE_USERNAME,
#      KAGGLE_KEY) read via kaggle_secrets.UserSecretsClient -- secrets are
#      NOT exported as environment variables by Kaggle, which is why the
#      persist script used to print "[persist] no credentials".
#
# On success exports both variables (so the kaggle CLI inherits them) and
# returns 0; otherwise returns 1.

kaggle_resolve_credentials() {
  if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    return 0
  fi
  local cfg="${HOME:-}/.kaggle/kaggle.json"
  if [ -f "$cfg" ]; then
    KAGGLE_USERNAME="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["username"])' "$cfg" 2>/dev/null || true)"
    KAGGLE_KEY="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["key"])' "$cfg" 2>/dev/null || true)"
    if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
      export KAGGLE_USERNAME KAGGLE_KEY
      echo "[creds] loaded from $cfg"
      return 0
    fi
  fi
  local pair
  pair="$(python - <<'PY' 2>/dev/null || true
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    print(secrets.get_secret("KAGGLE_USERNAME") or "")
    print(secrets.get_secret("KAGGLE_KEY") or "")
except Exception:
    pass
PY
)"
  local username key
  username="$(printf '%s\n' "$pair" | sed -n 1p)"
  key="$(printf '%s\n' "$pair" | sed -n 2p)"
  if [ -n "$username" ] && [ -n "$key" ]; then
    KAGGLE_USERNAME="$username"
    KAGGLE_KEY="$key"
    export KAGGLE_USERNAME KAGGLE_KEY
    echo "[creds] loaded from Kaggle notebook secrets"
    return 0
  fi
  return 1
}
