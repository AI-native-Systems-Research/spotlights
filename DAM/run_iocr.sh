nohup spotlights-engine \
  --repo ~/work/IOCR \
  --no-deep-research \
  --objective "reduce end-to-end page latency: wall-clock time from page image in to final OCR result out" \
  --hint "CPU-only deployment, no GPU available" \
  --hint "Mixed production traffic: business document scans - bills of lading, air waybills, purchase orders, executed MSAs, bank forms - including deliberately low-quality background scans" \
  --hint "Full-page 300 DPI scans, median 2550x3301 px (~8.4 MP), range 3.8-10.1 MP, dense English text, one page per request" \
  --hint "OCR accuracy must not regress; correctness is measured by the existing OCR-Benchmark harness" \
  --max-parallel 1 \
  --output-folder ./spotlights-out \
  --artifacts-dir ./artifacts \
  --verbose \
  --log-file ./spotlights-out/run.log \
  > ./spotlights-out/nohup.out 2>&1 &
