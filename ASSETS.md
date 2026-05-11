# Harness Alpha Local Asset Layer

Harness Alpha keeps physics and validation in Python/Pymunk. Visual assets are
renderer-only set dressing used by the Godot demo client.

## Local Packs

The current local library uses CC0 Kenney packs stored under:

`assets/library/kenney/`

Downloaded packs:

- `simple-space`
- `new-platformer-pack`
- `sports-pack`
- `foliage-pack`
- `background-elements`
- `particle-pack`

Each pack includes its own `License.txt`. The downloaded binary assets are
ignored by Git because they are large local/demo artifacts.

## Download/Refresh

To recreate the local library from official Kenney pages:

```powershell
.\venv\Scripts\python.exe download_assets.py
```

Use `--force` to redownload and reextract every pack.

## Indexing

Build or refresh the semantic index after downloading packs:

```powershell
.\venv\Scripts\python.exe asset_resolver.py build
```

The index is written to `assets/asset_index.json`. World exports use it to map
Visual Director hints such as `tree`, `spaceship`, `fire`, or `soccer` to local
PNG paths. Godot loads those paths as decorative props only; they never change
collision, objective state, validation, or reward truth.
