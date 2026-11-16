#!/bin/bash
# Double-click in Finder: tests HumansRUS, puts the training images in place,
# retrains the model and shows the results. Safe to run again; nothing is pushed.
cd "$(dirname "$0")" || exit 1
chmod +x humansrus.sh HumansRUS.command 2>/dev/null
say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
stop() { say "$1"; echo "Copy the last lines above and send them to Claude."; read -r -p "Press Enter to close. "; exit 1; }

say "1/4  Testing the website, checker and metrics (first run installs packages: 10-15 min)"
if [ -f .humansrus-tests-passed ]; then echo "(tests already passed earlier; skipping)"; else ./humansrus.sh test || stop "A test failed."; touch .humansrus-tests-passed; fi

say "2/4  Putting the training images in place"
mkdir -p data/matched
have=$(ls data/matched | grep -ci -E '\.(jpg|jpeg|png|webp)$')
if [ "$have" -lt 1200 ]; then
  shopt -s nullglob nocaseglob
  found=0
  # zips from Drive (any browser), in Downloads or on the Desktop
  for z in "$HOME"/Downloads/*matched*.zip "$HOME"/Downloads/drive-download*.zip "$HOME"/Desktop/*matched*.zip; do
    echo "unzipping $(basename "$z")"; unzip -o -j -q "$z" -d data/matched && found=1
  done
  # Safari unzips downloads by itself and leaves a folder instead
  for d in "$HOME"/Downloads/*matched*/ "$HOME"/Downloads/drive-download*/ "$HOME"/Desktop/*matched*/; do
    echo "copying images from $(basename "$d")"
    find "$d" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) -exec cp {} data/matched/ \; && found=1
  done
  if [ "$found" = 0 ]; then
    echo "Newest items in Downloads:"; ls -t "$HOME/Downloads" | head -15
    stop "Couldn't find the 'matched' download in Downloads (zip or folder)."
  fi
  have=$(ls data/matched | grep -ci -E '\.(jpg|jpeg|png|webp)$')
fi
echo "images in data/matched: $have"
[ "$have" -ge 1200 ] || stop "Expected 1200 training images but found $have."

say "3/4  Retraining the model (1-2 hours; keep the Mac plugged in and this window open)"
./humansrus.sh improve || stop "The retraining stopped with an error."

say "4/4  Finished. Results are in evaluation/results/improve/summary.md"
echo "Tell Claude \"done\" and it will check whether the new model is better."
read -r -p "Press Enter to close. "
