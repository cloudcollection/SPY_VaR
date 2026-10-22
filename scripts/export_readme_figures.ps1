param(
    [Parameter(Mandatory=$true)][string]$Deck,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [Parameter(Mandatory=$true)][string]$WorkDirectory
)
$ErrorActionPreference = 'Stop'
$deckPath = (Resolve-Path -LiteralPath $Deck).Path
New-Item -ItemType Directory -Force -Path $OutputDirectory,$WorkDirectory | Out-Null
$outputPath = (Resolve-Path -LiteralPath $OutputDirectory).Path
$env:TEMP = (Resolve-Path -LiteralPath $WorkDirectory).Path
$env:TMP = $env:TEMP
$renderPath = Join-Path $env:TEMP ('render-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $renderPath | Out-Null
$names = @('research_workflow','c2_architecture','patch_transformer','historical_evaluation')
$app = $null
$presentation = $null
$stage = 'Start PowerPoint'
try {
    $app = New-Object -ComObject PowerPoint.Application
    $stage = 'Open saved PPTX'
    $presentation = $app.Presentations.Open($deckPath, -1, 0, 0)
    if ($presentation.Slides.Count -ne $names.Count) { throw 'Unexpected number of figure slides.' }
    for ($i=1; $i -le $names.Count; $i++) {
        $stage = 'Export slide ' + $i
        $presentation.Slides.Item($i).Export((Join-Path $renderPath ($names[$i-1]+'.png')), 'PNG', 2400, 1350)
    }
    $presentation.Close()
    $presentation = $null
    $stage = 'Copy final PNG figures'
    foreach ($name in $names) {
        Copy-Item -LiteralPath (Join-Path $renderPath ($name+'.png')) -Destination (Join-Path $outputPath ($name+'.png')) -Force
    }
} catch {
    throw ($stage + ': ' + $_.Exception.Message)
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $app) {
        $app.Quit()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
    }
}
