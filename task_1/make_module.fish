#!/usr/bin/fish

# This fish script is for making modules importable by main.py - it can be safely ignored.

if ! test $argv[1] || ! test $argv[2]
	echo -e "Please include a folder to make a module!\n\tUsage:\t" (status -f) " module_folder filename\n\te.g. " (status -f) " cnn main"
	exit 1
end

echo -e "Making" (echo $argv) "a compatible python module..."

touch $argv[1]/__init__.py
echo -e "from . "(echo $argv[2]) " import fit, predict" > $argv[1]/__init__.py

echo "done"
