
Get-ChildItem -Path "C:\", "$env:USERPROFILE" -Filter "ggplot2" -Directory -Recurse -ErrorAction SilentlyContinue -Depth 5 | Select-Object FullName

