# AI Resume Fairness Auditor

A Python-based resume screening tool that evaluates resumes against a
job description (JD) using structured rule-based checks and semantic
similarity scoring. It produces ranked candidate shortlists and detailed
CSV/JSON reports through an interactive Streamlit dashboard.

> **Important fairness note:** The current implementation supports
> configurable screening rules and per-candidate explanations, but it
> does **not** yet implement formal statistical fairness metrics or
> bias-mitigation methods. Treat results as decision support for human
> reviewers, not as an automated hiring decision.

## Table of Contents

-   [Overview](#overview)
-   [Key Features](#key-features)
-   [Architecture](#architecture)
-   [Screening Workflow](#screening-workflow)
-   [Scoring and Candidate Status](#scoring-and-candidate-status)
-   [Project Structure](#project-structure)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Run the Application](#run-the-application)
-   [Run the Pipeline from the Command
    Line](#run-the-pipeline-from-the-command-line)
-   [Inputs and Outputs](#inputs-and-outputs)
-   [Current Limitations](#current-limitations)
-   [Roadmap](#roadmap)
-   [Responsible Use](#responsible-use)

## Overview

The **AI Resume Fairness Auditor** helps recruiters and data-science
teams screen a collection of resumes against a structured job
description. It cleans resume data, extracts candidate information,
evaluates candidates using configurable skills, education, and
experience gates, and compares resume text with the job description
using a Sentence-Transformers model.

The pipeline combines gate-based and semantic scores to produce a
shortlist, review, or rejection status. It also records matched and
missing skills and exports results for further review.

## Key Features

-   **Resume data cleaning:** Normalizes resume text and category
    labels.
-   **Resume parsing:** Extracts skills, education, experience, and
    skill-specific experience where available.
-   **Flexible job-description input:** Supports manual text, a
    job-description URL, saved job descriptions, and fallback sources.
-   **Configurable gate cascade:** Evaluates skills, education, and
    experience using ordered, weighted rules.
-   **Semantic similarity:** Uses the `all-MiniLM-L6-v2`
    Sentence-Transformers model to compare resume text with the selected
    job description.
-   **Combined scoring:** Combines gate and semantic scores using
    configurable weights.
-   **Candidate explanations:** Records matched/missing skills and
    gate-level reasons.
-   **CSV and JSON reports:** Produces tabular shortlist results and
    detailed cascade reports.
-   **Interactive dashboard:** Uses Streamlit to display screening
    results, summaries, charts, and download options.
-   **Saved job descriptions:** Supports a reusable job-description
    library in `saved_jds/`.

## Architecture

``` text
data/Resume.csv
      |
      v
src/data_cleaner.py
      |
      v
output/resumes_cleaned.csv
      |
      v
src/resume_parser.py ---- src/skill_extractor.py
      |
      +-----------------------+
                              |
Job description              |
(manual text / URL / saved)   |
      |                       |
      v                       v
src/jd_parser.py ------> src/pipeline.py
                              |
                 +------------+------------+
                 |                         |
                 v                         v
        src/gate_cascade.py       src/semantic_scorer.py
                 |                         |
                 +------------+------------+
                              |
                              v
                    Combined candidate scores
                              |
                              v
             output/shortlist_results.csv / .json
                              |
                              v
                    app/streamlit_app.py
```

Shared paths, thresholds, and scoring configuration are defined in
`src/config.py`.

## Screening Workflow

1.  **Clean resume data:** `data_cleaner.load_and_clean_resumes()` reads
    `data/Resume.csv`, cleans resume text and category labels, and
    writes `output/resumes_cleaned.csv`.
2.  **Parse resumes:** `resume_parser.parse_all_resumes()` creates
    structured candidate profiles using the resume parsing and
    skill-extraction modules.
3.  **Provide a job description:** The pipeline can accept manually
    entered text, scrape a job-description URL, or load a saved JD. It
    may also use `data/postings.csv` or a configured fallback JD.
4.  **Review the JD:** The pipeline displays/logs the parsed job
    description for review.
5.  **Evaluate rule-based gates:**
    `gate_cascade.run_candidate_cascade()` evaluates skills, education,
    and experience using configurable gate rules and weights.
6.  **Calculate semantic similarity:**
    `SemanticScorer.compute_similarity_scores()` embeds the job
    description and resumes with `all-MiniLM-L6-v2` and calculates
    cosine similarity scores.
7.  **Combine scores:** `calculate_semantic_combined_score()` combines
    gate and semantic scores.
8.  **Assign candidate status:** Configured thresholds determine whether
    a candidate is shortlisted, marked for review, or rejected. Gate
    failures may trigger early rejection when `stop_on_failure` is
    enabled.
9.  **Export results:** The pipeline saves CSV and JSON reports and logs
    summary information, including top candidates and funnel counts.

## Scoring and Candidate Status

The overview supplied for this project describes these default values:

  --------------------------------------------------------------------------
  Setting                                      Default Meaning
  ----------------------- ---------------------------- ---------------------
  Gate score weight                               0.70 Contribution of the
                                                       rule-based gate score

  Semantic score weight                           0.30 Contribution of
                                                       semantic similarity

  `SHORTLIST_THRESHOLD`                             70 Scores at or above
                                                       this threshold are
                                                       shortlisted, subject
                                                       to gate logic

  `MAYBE_THRESHOLD`                                 50 Scores at or above
                                                       this threshold may be
                                                       marked for review,
                                                       subject to gate logic
  --------------------------------------------------------------------------

The combined score is described as:

``` text
final_score = (gate_score * 0.70) + (semantic_score * 0.30)
```

The actual values and status logic should be verified in
`src/config.py`, `src/semantic_scorer.py`, and `src/pipeline.py` before
changing or relying on them. The semantic model returns similarity
percentages; ensure both score inputs use compatible scales.

## Project Structure

``` text
AI-Resume-Fairness-Auditor/
├── app/
│   └── streamlit_app.py       # Interactive Streamlit application
├── data/
│   ├── Resume.csv             # Source resume dataset
│   └── postings.csv           # Optional job-posting fallback, if present
├── output/
│   ├── resumes_cleaned.csv    # Cleaned resume data
│   ├── shortlist_results.csv  # Tabular screening results
│   ├── shortlist_results.json # Detailed cascade reports
│   ├── eda_charts/            # Exploratory plots, when generated
│   └── sample_parsed_resumes.json # Debug/sample parsed profiles, if generated
├── saved_jds/                 # Saved job descriptions, if present
├── src/
│   ├── config.py              # Shared paths, thresholds, weights
│   ├── data_cleaner.py        # Resume data cleaning
│   ├── resume_parser.py       # Structured resume parsing
│   ├── skill_extractor.py     # Skill, education, and experience extraction
│   ├── jd_parser.py           # JD parsing, URL scraping, saved JD handling
│   ├── gate_cascade.py        # Rule-based gate evaluation
│   ├── semantic_scorer.py     # Embedding similarity and score combination
│   ├── pipeline.py            # End-to-end screening orchestration
│   ├── full_pipeline.py       # CLI convenience wrapper
│   ├── eda_analytics.py       # Exploratory analysis utilities
│   ├── model_comparator.py    # Model comparison utilities
│   └── score_explainer.py     # Score explanation utilities
├── requirements.txt           # Python dependencies
└── README.md
```

This is a logical overview based on the project description. Some files
and generated outputs may be absent from a particular checkout; confirm
the actual repository contents before relying on every entry above.

## Requirements

-   Python installed and available from the terminal.
-   Git, if cloning the project from GitHub.
-   Python dependencies listed in `requirements.txt`.
-   Internet access may be required on the first run to download the
    Sentence-Transformers model or other resources, unless they are
    already cached.
-   The source resume dataset at `data/Resume.csv`.

The project overview lists dependencies including pandas, spaCy,
Sentence-Transformers, PyTorch, Beautiful Soup, Requests, and Streamlit.
The exact dependency versions are defined in `requirements.txt`.

## Installation

Run the following commands from the project root.

### 1. Clone the repository

Replace `<repository-url>` with your GitHub repository URL.

``` bash
git clone <repository-url>
cd AI-Resume-Fairness-Auditor
```

If you already have the project locally, open its root folder instead.

### 2. Create a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use your environment's approved
activation method or run the virtual environment's Python executable
directly.

**macOS/Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Application

From the project root, with the virtual environment activated, run:

``` bash
python -m streamlit run app/streamlit_app.py
```

Streamlit prints a local address in the terminal, typically:

``` text
http://localhost:8501
```

Open that address in your browser. Use the dashboard to provide or
select a job description, run screening, review results, and download
the available CSV/JSON reports.

To stop the local server, press `Ctrl+C` in the terminal.

## Run the Pipeline from the Command Line

The project overview identifies `src/full_pipeline.py` as a convenience
wrapper:

``` bash
python src/full_pipeline.py
```

The pipeline can also be invoked programmatically. For example, from the
project root:

``` bash
python -c "from src.pipeline import run_screening_pipeline; run_screening_pipeline(mode='text', text='Paste the job description here')"
```

Confirm the accepted arguments in `src/pipeline.py` before using this
example; CLI behavior and available options may vary with the current
implementation.

To clean the source resume dataset manually:

``` bash
python src/data_cleaner.py
```

Run data cleaning only when needed, and keep a backup of source data
before making changes.

## Inputs and Outputs

### Inputs

  -----------------------------------------------------------------------
  Path                                Purpose
  ----------------------------------- -----------------------------------
  `data/Resume.csv`                   Source resume dataset; the project
                                      overview reports approximately
                                      2,484 rows

  `data/postings.csv`                 Optional job-posting fallback, if
                                      available

  Manual job-description text         User-provided role requirements

  Job-description URL                 Optional source to scrape and parse

  `saved_jds/`                        Saved job descriptions that can be
                                      reused
  -----------------------------------------------------------------------

### Outputs

  -------------------------------------------------------------------------
  Path                                  Purpose
  ------------------------------------- -----------------------------------
  `output/resumes_cleaned.csv`          Cleaned resume text, categories,
                                        and related fields

  `output/shortlist_results.csv`        Candidate ranks, scores, status,
                                        rejection reason, matched skills,
                                        and missing skills

  `output/shortlist_results.json`       Detailed per-candidate cascade
                                        reports and timestamps

  `output/eda_charts/`                  Exploratory charts when the
                                        analytics utilities are run

  `output/sample_parsed_resumes.json`   Sample parsed profiles for
                                        debugging, if generated
  -------------------------------------------------------------------------

The precise columns depend on the current code and configuration.

## Current Limitations

The current project overview identifies the following limitations:

-   **Formal fairness metrics are not implemented.** Metrics such as
    demographic parity, equal opportunity, or disparate impact are not
    currently calculated by the described default pipeline.
-   **Bias mitigation is not implemented.** The project does not
    currently include a verified statistical mitigation procedure.
-   **Protected-attribute analysis is absent.** Do not infer or derive
    sensitive characteristics from names, photos, or resume text for
    employment decisions.
-   **Explainability is partial.** Gate-level reasons and
    matched/missing skills are available, but global feature-importance
    or SHAP-based explanations are not described as integrated.
-   **Supporting analytics are not integrated by default.**
    `eda_analytics.py`, `model_comparator.py`, and `score_explainer.py`
    are described as utilities outside the default pipeline.
-   **Gate weight validation is needed.** The project overview notes
    that JD-specific gate weights are not validated to ensure they sum
    to one.
-   **Fallback job descriptions may be unsuitable.** Verify the selected
    JD matches the intended role before screening.
-   **Model and environment issues may occur.** The semantic model may
    need to be downloaded on first use; repeated runs in a long-lived
    Streamlit session may need memory monitoring.
-   **Test coverage needs improvement.** Edge cases for parsing, gate
    evaluation, semantic scoring, and output generation should be
    tested.

These are findings from the supplied project overview, not a fresh code
audit.

## Roadmap

Recommended next development steps, in priority order:

1.  **Define and implement a responsible fairness evaluation
    methodology.** Use appropriately collected, legally permissible data
    and document the limitations of any group-level evaluation.
2.  **Evaluate potential mitigation methods.** Test any proposed
    intervention against accuracy, utility, and group-level outcomes
    before claiming that it reduces bias.
3.  **Refactor imports and package structure.** Standardize module
    imports and define a clean public API.
4.  **Add automated tests.** Cover JD parsing, skill extraction, gate
    scoring, semantic scoring, thresholds, and error handling.
5.  **Improve the dashboard.** Add clearly labeled evaluation results,
    threshold sensitivity analysis, and appropriate audit summaries once
    valid fairness measures are implemented.

## Responsible Use

This tool is intended to support, not replace, human review.

-   Verify that job criteria are relevant to the role and distinguish
    genuine requirements from preferences.
-   Review rejected and shortlisted candidates before taking
    employment-related action.
-   Do not treat semantic similarity as proof of qualification or job
    performance.
-   Check data quality, accessibility, and the potential impact of
    screening rules on different groups.
-   Follow applicable employment, privacy, and data-protection laws and
    organizational policies.
-   Do not claim that the system is "bias-free" or that it audits
    fairness quantitatively until those capabilities have been
    implemented and validated.

## Contributing

Contributions are welcome. Useful areas include test coverage, clearer
configuration validation, robust parsing, documentation, and responsible
fairness evaluation.

Before submitting changes:

1.  Explain the problem or improvement.
2.  Add or update tests where practical.
3.  Document configuration or behavior changes.
4.  Avoid committing private candidate data, API keys, credentials, or
    generated files containing personal information.

## License

No license information was included in the supplied project overview.
Add a `LICENSE` file and update this section before publishing the
repository if you intend to specify reuse permissions.
