# CI Test Reports

This page summarizes JUnit outputs under `artifacts/test-reports/` and embeds the latest **pytest-html** report when present (copied to `_static/pytest-report.html` by `docs/scripts/generate_ci_test_report.py` before the book build).

## Embedded HTML report

The report is shown inline so you do not leave the documentation layout. If no `report.html` was found when the book was built, a short placeholder page appears instead.

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report.html" width="100%" height="1200"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML report"></iframe>
   </div>
```

## JUnit summary

```{include} ../_generated/test_report_summary.md
```
