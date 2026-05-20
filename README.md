# Lica Score Web Report

Open `index.html` directly, or run:

```bash
cd /home/ubuntu/lica-score-web
python -m http.server 8000
```

Then open http://localhost:8000.

## Regenerate

Large model caches are routed to `/mnt/local` by `cache_config.py`.

```bash
python generate_report.py
python add_baselines.py --keep-going
```
