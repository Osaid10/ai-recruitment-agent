"""Years of experience is computed from dates, so the date reading has to be right.

Two bugs here were found by running real input rather than by reasoning about
the code:

1. The model undercounted every candidate by 2.4-3.0 years, because it reads
   "Present" as its training cutoff. Fixed by making the date arithmetic
   authoritative — see extract.extract_profile.
2. The date arithmetic then counted a degree's "2022 - 2026" as employment,
   crediting a final-year undergraduate with 4.9 years of professional
   experience — enough to clear a senior role's gate. Found by running a real
   student CV through the pipeline.

Both directions matter: over-counting waves through someone unqualified,
under-counting silently rejects someone who is.
"""

from __future__ import annotations

from datetime import date

from recruiter.ingest.extract import heuristic_years

TODAY = date(2026, 8, 6)


# -- ordinary employment ---------------------------------------------------


def test_a_single_dated_role_is_counted() -> None:
    # Jan 2020 to Jan 2022 is two years exactly — the span is measured, not
    # counted inclusively at both ends.
    assert heuristic_years("Engineer\nJan 2020 - Jan 2022", TODAY) == 2.0


def test_present_means_today_not_the_models_cutoff() -> None:
    years = heuristic_years("Senior Engineer\nAug 2019 - Present", TODAY)
    assert 6.9 < years < 7.1, years


def test_overlapping_roles_are_merged_not_double_counted() -> None:
    """Two concurrent roles are not twice the experience."""
    text = "Engineer\nJan 2020 - Jan 2023\nConsultant (concurrent)\nJan 2021 - Jan 2022"
    assert heuristic_years(text, TODAY) == 3.0  # not 4.0


def test_a_gap_between_roles_is_not_counted_as_worked_time() -> None:
    continuous = "Role A\nJan 2018 - Jan 2020\nRole B\nJan 2020 - Jan 2022"
    with_gap = "Role A\nJan 2018 - Jan 2020\nRole B\nJan 2021 - Jan 2022"
    assert heuristic_years(with_gap, TODAY) < heuristic_years(continuous, TODAY)


# -- the shared-year form, previously missed entirely ----------------------


def test_month_to_month_with_one_shared_year_is_counted() -> None:
    """'Jun - Aug 2024' is how most resumes write a short internship.

    The original pattern required a year on the left of the dash, so this
    matched nothing and the role was invisible.
    """
    years = heuristic_years("AI Intern | Example Corp\nJun - Aug 2024", TODAY)
    assert years > 0, "a real internship was not counted at all"
    assert years < 0.5, years


def test_shared_year_range_running_to_present() -> None:
    years = heuristic_years("Engineer\nMar - Present 2026", TODAY)
    assert years >= 0


# -- education must not count as employment -------------------------------


def test_degree_dates_are_not_professional_experience() -> None:
    """The bug a real student CV exposed."""
    cv = (
        "EDUCATION\n"
        "BS Artificial Intelligence | Some Institute of "
        "Engineering Sciences and Technology 2022 - 2026\n"
    )
    assert heuristic_years(cv, TODAY) == 0.0


def test_a_student_cv_counts_only_the_internship() -> None:
    cv = (
        "EDUCATION\n"
        "BS Artificial Intelligence | Some Institute of Technology 2022 - 2026\n"
        "PROFESSIONAL EXPERIENCE\n"
        "AI Intern | Example Corp, Islamabad Jun - Aug 2024\n"
    )
    years = heuristic_years(cv, TODAY)
    assert 0.0 < years < 0.5, f"expected roughly one summer, got {years}"


def test_student_society_roles_are_not_professional_experience() -> None:
    cv = "Event Coordinator | Campus Art Society, Some Institute 2025 - 2026\n"
    assert heuristic_years(cv, TODAY) == 0.0


def test_education_exclusion_does_not_swallow_real_roles() -> None:
    """The exclusion is per line — it must not drop the job above or below it."""
    cv = (
        "Senior Engineer | Acme\nJan 2020 - Jan 2024\n"
        "EDUCATION\nBSc Computer Science, Some University 2014 - 2018\n"
        "Engineer | Other Co\nJan 2018 - Jan 2020\n"
    )
    years = heuristic_years(cv, TODAY)
    assert 5.9 < years < 6.3, f"expected the two jobs only, got {years}"


# -- degenerate input ------------------------------------------------------


def test_no_dates_returns_zero_rather_than_guessing() -> None:
    assert heuristic_years("Python developer. Available immediately.", TODAY) == 0.0


def test_empty_text_is_zero() -> None:
    assert heuristic_years("", TODAY) == 0.0


def test_a_reversed_range_is_ignored() -> None:
    assert heuristic_years("Engineer\nJan 2022 - Jan 2020", TODAY) == 0.0
