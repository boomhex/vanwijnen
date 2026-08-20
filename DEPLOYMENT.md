# Deploying on a new Linux machine

Everything the app needs lives in gitignored files that only exist on
machines someone has manually set up (`.env`, `users.json`, `config.json`,
`.storage_secret`). This is the checklist to reconstruct all of it on a new
box without guessing. Commands assume the repo is cloned to `/opt/vanwijnen`
and run as root/sudo unless noted - adjust the path if you use a different
location (and update it in `development/app/deploy/vanwijnen-offerapp.service`
to match).

## 1. Prerequisites

- Python 3.11+ (developed and tested on 3.13)
- git

## 2. Get the code and a service account

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin vanwijnen
sudo git clone <repo-url> /opt/vanwijnen
sudo chown -R vanwijnen:vanwijnen /opt/vanwijnen
```

## 3. Python environment

```bash
cd /opt/vanwijnen
sudo -u vanwijnen python3 -m venv venv
sudo -u vanwijnen venv/bin/pip install -r requirements.txt
```

`requirements.txt` is pinned to versions that are actually tested against
this codebase - notably `google-genai`, `python-dotenv`, and
`nicegui-tabulator`, which are imported by the app but were missing from
the file until this pass (a fresh install from the old file would have
crashed on startup). Use `requirements-dev.txt` instead if you also want
to run the test suite (`pytest`).

## 4. Secrets and config

```bash
cd /opt/vanwijnen/development/app
sudo -u vanwijnen cp .env.example .env
sudo -u vanwijnen $EDITOR .env   # fill in GEMINI_API_KEY
```

- `GEMINI_API_KEY` is the only required value - see `.env.example` for the
  optional ones (`PORT`, `STORAGE_SECRET`).
- `config.json` is optional; if absent the app falls back to built-in
  defaults (e.g. a 120s LLM request timeout). Only create one if you need
  to override something - see `services/extract_offer.py`'s `load_config()`.
- `.storage_secret` (signs session cookies) is auto-created on first run if
  `STORAGE_SECRET` isn't set in `.env` - no action needed.

## 5. First user

```bash
cd /opt/vanwijnen/development/app
sudo -u vanwijnen env PYTHONPATH=. ../../venv/bin/python -m services.auth add <username> <password>
sudo -u vanwijnen env PYTHONPATH=. ../../venv/bin/python -m services.auth grant <username> <workspace>
```

`<workspace>` is just a folder name under `storage/` - it gets created the
first time something is uploaded to it. Repeat `add`/`grant` for each
teammate; see `services/auth.py` for `remove`/`revoke`/`list`.

## 6. Run it as a service

The app has no built-in process supervision - left running as a bare
`python main.py`, a crash or reboot takes it down until someone notices.
`development/app/deploy/vanwijnen-offerapp.service` is a systemd unit that
restarts it automatically and starts it at boot.

```bash
sudo cp development/app/deploy/vanwijnen-offerapp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vanwijnen-offerapp
sudo systemctl status vanwijnen-offerapp
```

Open the unit file first and check the `WorkingDirectory` /
`EnvironmentFile` / `ExecStart` paths and the `User`/`Group` match what you
actually used above.

## 7. Verify

```bash
curl -sf -o /dev/null -w '%{http_code}\n' http://localhost:8080/login   # expect 200
journalctl -u vanwijnen-offerapp -f                                     # service logs
tail -f /opt/vanwijnen/development/app/logs/app.log                     # app-level action log
```

## Updating the app later

`git pull` inside `/opt/vanwijnen` is picked up automatically within a few
seconds - `main.py` runs with hot-reload enabled (`uvicorn_reload_dirs`),
so most code changes don't need a restart. You only need
`sudo systemctl restart vanwijnen-offerapp` after:

- a `requirements.txt` change (re-run `pip install -r requirements.txt` first)
- an edit to the `ui.run(...)` call in `main.py` itself (its config is
  fixed at process start and isn't picked up by hot-reload)
