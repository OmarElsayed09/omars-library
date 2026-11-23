$ErrorActionPreference = 'Stop'
$dest = Join-Path $PSScriptRoot 'lectures'
New-Item -Path $dest -ItemType Directory -Force | Out-Null
$assets = @(
    '50702172c7cdc969615b81e6f22499fd_MIT18_06S10_L01.pdf',
    'da601f10fd86ba00d14b378327561b84_MIT18_06S10_L02.pdf',
    '18d1a0a50c5d91ead6f30195cc161f20_MIT18_06S10_L03.pdf'
)
foreach ($asset in $assets) {
    $filename = $asset.Substring(33)
    Invoke-WebRequest "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/$asset" -OutFile (Join-Path $dest $filename)
}
