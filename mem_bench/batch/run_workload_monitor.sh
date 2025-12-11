#!/bin/bash
# Application-level GPU Power Measurement using AMD-SMI
# Usage ./power_measure.sh "amd-smi ..." "rocm-smi ..." "rocm-smi ..." "application ..."

echo "Starting monitoring..."
($1) &
amdsmipid=$!
($2) &
rocmpid=$!
echo "Sleeping for a while..."
sleep 10
echo "Launching application..."
($3) &
appid=$!
wait "$appid"
echo "Application exited, sleeping for a while..."
sleep 10
echo "Ending monitoring..."
kill "$amdsmipid"
kill "$rocmpid"
echo "Making sure CSV dump is completed, sleeping again..."
sleep 60