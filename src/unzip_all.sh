#!/bin/bash
# unzip_all.sh — recursively unzip all .zip files and remove them

set -e

find "$1" -type f -name "*.zip" | while IFS= read -r zipfile; do

  zipdir=$(dirname "$zipfile")
  echo "Unzipping: $zipfile"

  if unzip -q "$zipfile" -d "$zipdir"; then
    echo "Removing: $zipfile"
    # rm "$zipfile"
  else
    echo "Failed to unzip $zipfile — keeping the archive."
  fi
done

echo "Done processing all zip files."
