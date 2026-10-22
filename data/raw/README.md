3# GraphiSense Dataset

The handwriting dataset is not included in this repository. Download it from the
[Mendeley Data record](https://data.mendeley.com/datasets/39hr8dx76p/1), *Potential
Dysgraphia Handwriting Dataset of School-Age Children* (Ramlan et al., Oct 2023,
DOI: [10.17632/39hr8dx76p.1](https://data.mendeley.com/datasets/39hr8dx76p.1)).
Follow the provider's license and attribution requirements.

Extract the dataset so the two class folders are directly under `data\raw`:

```text
data\raw\
├── Potential Dysgraphia\
└── Low Potential Dysgraphia\
```

For example, from the repository root in PowerShell:

```powershell
$datasetZip = Read-Host "Enter the full path to the downloaded Mendeley dataset ZIP"
$datasetRoot = Join-Path $PWD "data\raw"
if (-not (Test-Path -LiteralPath $datasetZip -PathType Leaf)) { throw "Dataset ZIP not found: $datasetZip" }
New-Item -ItemType Directory -Force -Path $datasetRoot
Expand-Archive -LiteralPath $datasetZip -DestinationPath $datasetRoot
Get-ChildItem -Path $datasetRoot
```

If the ZIP contains an additional enclosing directory, move the two class folders
directly into `$datasetRoot`. The raw dataset directory is excluded from Git; do
not commit the downloaded images.
