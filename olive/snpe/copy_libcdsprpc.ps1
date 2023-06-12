# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# copies the libcdsprpc.dll from driver location to given location
Set-PSDebug -Trace 2
if ( $args.count -eq 0 ) {
  echo "Please specify the output location of libcdsprpc.dll"
  exit 1
}
$loc = [string](driverquery /v /fo csv | findstr qcadsprpc)
if ( $loc -eq $null ) {
  driverquery /v /fo csv
  echo "Cannot locate FastRPC driver"
  exit 1
}
$lll2 = $loc.Split(",")[15]
if ( $lll2 -eq $null ) {
  echo "Cannot locate path from FastRPC driver query"
  exit 1
}
$lll = $lll2.Split('"')[1]
if ( $lll -eq $null ) {
  echo "Cannot locate path from FastRPC driver query"
  exit 1
}
echo Driver location is: $lll
$dir = Split-Path $lll
# $dir = [System.IO.Path]::GetDirectoryName($lll)
$f = Join-Path $dir -ChildPath libcdsprpc.dll
echo Copying $f to $args[0]
Copy-Item -Path $f -Destination $args[0]
