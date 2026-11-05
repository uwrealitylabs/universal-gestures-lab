# Dataset inventory

Run from the repository root:

```powershell
python tools/dataset_inventory.py "src/data(Dynamic)" --report inventory-report.json
python tools/dataset_inventory.py --self-test
python -m unittest discover -s tools -p "test_dataset_inventory.py"
```

When no report path is supplied, the tool chooses an unused `dataset_inventory_report` filename. An explicit report path must not already exist. Reports cannot be placed inside the scanned directory or overwrite existing files. The input recordings are read only. The optional `--opposition-index N` replaces the default opposition indexes 5, 9, 13 and 16, based on `GetOneHandData` in the Unity repository's `Assets/Scripts/UGDataExtractorScript.cs`.

Each file must contain an object or a list of recording objects. An empty list or empty object is a warning. Other recordings require `sequenceData` and numeric binary `confidence`; extra fields are allowed. A sequence must contain frames; each frame must be a list of 17 finite numeric values (booleans are excluded). A nonempty sequence shorter than 15 frames and negative opposition distances generate warnings. The current multiclass loader chooses a window size from the data, so a short sequence is not necessarily dropped in every configuration.

Statistics are grouped by the immediate containing folder relative to the scan root. File counts include unreadable files. Recording counts include all entries in parsed lists and single objects, including invalid entries; unreadable files and other top-level types contribute none. Feature ranges include finite values in correctly sized frames, even if another value in that frame is invalid. Indexes start at zero; source line numbers start at one. Missing fields use the containing object's line. Errors include JSON syntax and validation failures; warnings describe unusual or incomplete data. The terminal displays at most five issue details, while the JSON report contains all issues.
