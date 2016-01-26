if ($env:appveyor_repo_tag -eq "True") {
  (Get-Content ci\.pypirc) | Foreach-Object {$_ -replace '%PASS%',$env:PYPI_PASS} | Set-Content $env:userprofile\.pypirc
  Invoke-Expression "$env:CMD_IN_ENV python setup.py bdist_wheel upload"
}
