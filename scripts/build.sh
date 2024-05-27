#!/bin/bash

cd ..
rm -rf public
rm -rf _output-public
rm -rf _output-private
rm -rf _output-presentation
source .venv/bin/activate

echo "BUILD public"
quarto render  --profile public
echo "BUILD private"
quarto render  --profile private
echo "BUILD presentation"
cd docs/presentation
quarto render
cd ../..
echo "COPY public into private"
cd _output-private/
cp -r ../_output-public/ .
cd ..
echo "COPY presentation"
cp -r _output-presentation _output-public/
cp -r _output-presentation _output-private/

echo "FINISHED"
