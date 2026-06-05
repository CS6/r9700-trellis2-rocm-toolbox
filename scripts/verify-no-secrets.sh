#!/usr/bin/env bash
set -euo pipefail

patterns=(
  'hf_[A-Za-z0-9]{20,}'
  '192\.168\.'
  '10\.[0-9]+\.[0-9]+\.'
  '172\.(1[6-9]|2[0-9]|3[0-1])\.'
  '/Users/'
  '/home/[A-Za-z0-9_-]+'
  'BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY'
)

for pattern in "${patterns[@]}"; do
  if grep -RInE "$pattern" . --exclude-dir=.git --exclude='verify-no-secrets.sh'; then
    echo "Potential secret/private value matched: $pattern" >&2
    exit 1
  fi
done

echo "No obvious secrets matched."
