# New machine setup and private-artifact restore

GitHub contains the public project-owned source and authorized showcase assets.
The USB backup contains the complete private research state: frozen models,
policies, raw results, internal archives, and visualization work.

## 1. Clone and install

Python 3.11 is the validated interpreter version.

```bash
git clone https://github.com/Eurekauro/franka-dynamics-priors-residual-sac.git
cd franka-dynamics-priors-residual-sac
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install pytest
python -m pip install -e . --no-deps
PYTHONPATH=src python tools/public_preflight.py
PYTHONPATH=src python -m pytest -q
```

## 2. Verify the USB backup

The backup directory is `Franka_2024_Full_Backup_20261024`. Replace
`<usb-drive-letter>` with the new Windows-assigned letter.

```bash
BACKUP=/mnt/<usb-drive-letter>/Franka_2024_Full_Backup_20261024
cd "$BACKUP"
sha256sum -c ARCHIVE_SHA256SUMS.txt
git bundle verify Franka_2024_Git_All_Refs.bundle
```

If WSL has not auto-mounted the drive, access it from Windows or mount the
confirmed removable volume. Do not assume the drive letter will remain `E:`.

## 3. Restore private artifacts into the clone

From the repository root, extract the owner-retained roots. This preserves the
newest public Git files while restoring private data:

```bash
tar -xzf "$BACKUP/Franka_2024_Full_Project.tar.gz" -C . \
  ./models ./policies ./results ./internal_archive ./visualization
```

The archive preserves Linux paths, modes, and link metadata despite being
stored on FAT32. It excludes `.git`, virtual environments, pytest/Python
caches, and bytecode.

Validate without training or rerunning P4/P5:

```bash
PYTHONPATH=src python tools/verify_migration.py
PYTHONPATH=src python tools/recompute_saved_results.py
find policies -type f -name best_model.zip | wc -l  # expected: 30
find models/priors -maxdepth 1 -type f -name '*.npz' | wc -l  # expected: 3
```

For a standalone restoration instead of a GitHub clone:

```bash
mkdir franka-dynamics-priors-residual-sac
tar -xzf "$BACKUP/Franka_2024_Full_Project.tar.gz" \
  -C franka-dynamics-priors-residual-sac
git clone "$BACKUP/Franka_2024_Git_All_Refs.bundle" franka-git-history
```

The Git bundle records full history at initial backup time. GitHub remains the
source for commits made after that archive.

## Scope and safety

The private backup is not a redistribution license. Do not publish frozen
weights, checkpoints, raw episodes, internal reviews, or third-party assets
beyond `NOTICE`, `docs/THIRD_PARTY_NOTICES.md`, and
`docs/LICENSING_AND_REUSE.md`. These checks do not establish real-robot safety
or certification.
