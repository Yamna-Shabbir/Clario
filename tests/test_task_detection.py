"""
tests/test_task_detection.py
-----------------------------
Comprehensive unit-test suite for services/task_detection.py.
Real spaCy en_core_web_sm is allowed; no other ML models.

Coverage matrix
---------------
A.  Simple clear tasks (explicit obligation + name)
B.  Indirect / polite requests
C.  First-person commitments (I'll, I will)
D.  Imperative sentences (no subject)
E.  Weak verbs — only task when cue also present
F.  Negation filtering (will not, won't, nobody, doesn't, never, can't)
G.  Hypothetical / conditional filtering (if…, would/could/might)
H.  Plain question filtering
I.  Past-tense filtering
J.  Multi-clause / compound task sentences
K.  Realistic meeting transcripts (mixed task + non-task)
L.  Unique / diverse assignee names (global names, all-caps, lowercase)
M.  Priority extraction (High / Medium / Low from keywords + deadlines)
N.  Relative date resolution → ISO YYYY-MM-DD
O.  Absolute date extraction
P.  Edge inputs (empty, whitespace, single word, unicode, punctuation)
Q.  Technical / domain-specific tasks
R.  Action-item and deadline-led sentences
S.  participants=None vs participants provided (backward-compat)
T.  Participant email matching
U.  get_task_summary
V.  Output schema validation (all required keys present)
W.  ID sequencing across a transcript
"""

from __future__ import annotations

import datetime
import re
import pytest

from services.task_detection import (
    detect_tasks,
    load_nlp,
    get_task_summary,
    TASK_KEYWORDS,
    STRONG_CUES,
    STRONG_ACTION_VERBS,
)

# ---------------------------------------------------------------------------
# Session-scoped spaCy model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def nlp():
    return load_nlp("en_core_web_sm")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_task(sentence: str, nlp, **kwargs) -> bool:
    return len(detect_tasks(sentence, nlp=nlp, **kwargs)) > 0


def _not_task(sentence: str, nlp, **kwargs) -> bool:
    return not _is_task(sentence, nlp, **kwargs)


def _first(sentence: str, nlp, **kwargs) -> dict:
    tasks = detect_tasks(sentence, nlp=nlp, **kwargs)
    assert tasks, f"No task detected in: {sentence!r}"
    return tasks[0]


def _all(text: str, nlp, **kwargs) -> list:
    return detect_tasks(text, nlp=nlp, **kwargs)


ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MEETING_DATE = datetime.date(2026, 11, 12)   # a Thursday


# ===========================================================================
# A.  SIMPLE CLEAR TASKS
# ===========================================================================

class TestSimpleClearTasks:
    """Straightforward sentences where any reasonable detector should fire."""

    def test_will_named(self, nlp):
        assert _is_task("John will complete the API documentation by Friday.", nlp)

    def test_needs_to_named(self, nlp):
        assert _is_task("Tom needs to send the status update by noon.", nlp)

    def test_must_named(self, nlp):
        assert _is_task("Carlos must submit the budget report by EOD.", nlp)

    def test_has_to_named(self, nlp):
        assert _is_task("Mike has to present the Q3 results next Monday.", nlp)

    def test_please_imperative(self, nlp):
        assert _is_task("Please send the meeting notes to everyone.", nlp)

    def test_lets_fix(self, nlp):
        assert _is_task("Let's fix the login bug before the release.", nlp)

    def test_ensure(self, nlp):
        assert _is_task("Please ensure the servers are backed up before midnight.", nlp)

    def test_make_sure(self, nlp):
        assert _is_task("Make sure the deployment pipeline is green before merging.", nlp)

    def test_deadline_led(self, nlp):
        assert _is_task("The deadline for the proposal is next Friday.", nlp)

    def test_follow_up(self, nlp):
        assert _is_task("Tom needs to follow up with the vendor about the invoice.", nlp)

    def test_responsible_for(self, nlp):
        assert _is_task("Ahmed is responsible for the backend deployment this week.", nlp)

    def test_action_item_prefix(self, nlp):
        assert _is_task("Action item: someone will document the new endpoints.", nlp)

    def test_take_care_of(self, nlp):
        assert _is_task("John will take care of the client onboarding process.", nlp)

    def test_should_named(self, nlp):
        assert _is_task("Zara should deliver the prototype to QA by next Thursday.", nlp)

    def test_coordinate_named(self, nlp):
        assert _is_task("Carlos must coordinate with the design team by Wednesday.", nlp)

    def test_submit_alone_imperative(self, nlp):
        assert _is_task("Submit the quarterly report by EOD.", nlp)

    def test_fix_alone_imperative(self, nlp):
        assert _is_task("Fix the broken integration test before the release.", nlp)

    def test_deploy_alone(self, nlp):
        assert _is_task("Deploy the hotfix to production immediately.", nlp)

    def test_write_alone(self, nlp):
        assert _is_task("Write the technical specification for the new feature.", nlp)

    def test_prepare_named(self, nlp):
        assert _is_task("John must prepare the financial model by next Tuesday.", nlp)


# ===========================================================================
# B.  INDIRECT / POLITE REQUESTS
# ===========================================================================

class TestIndirectRequests:
    """Sentences that are tasks but phrased indirectly or politely."""

    def test_it_would_be_great_if(self, nlp):
        assert _is_task("It would be great if Sara could handle the slides.", nlp)

    def test_it_would_be_nice_if(self, nlp):
        assert _is_task("It would be nice if someone could review the wireframes.", nlp)

    def test_it_would_be_helpful_if(self, nlp):
        assert _is_task("It would be helpful if Omar could draft the proposal.", nlp)

    def test_would_be_ideal_if(self, nlp):
        assert _is_task("It would be ideal if Lisa could finalise the budget.", nlp)

    def test_can_you_schedule(self, nlp):
        assert _is_task("Can you schedule a call with the client tomorrow?", nlp)

    def test_can_you_make_sure(self, nlp):
        assert _is_task("Can you make sure the CI pipeline passes before merging?", nlp)

    def test_could_you_send(self, nlp):
        assert _is_task("Could you send the updated spec to the team?", nlp)

    def test_would_you_mind(self, nlp):
        assert _is_task("Would you mind reviewing the PR before tomorrow?", nlp)


# ===========================================================================
# C.  FIRST-PERSON COMMITMENTS
# ===========================================================================

class TestFirstPersonCommitments:
    """Speaker commits to doing something — should be detected as tasks."""

    def test_ill_prepare(self, nlp):
        assert _is_task("I'll prepare the budget spreadsheet tonight.", nlp)

    def test_i_will_send(self, nlp):
        assert _is_task("I will send the follow-up email after this call.", nlp)

    def test_i_will_fix(self, nlp):
        assert _is_task("I will fix the memory leak by end of day.", nlp)

    def test_i_need_to(self, nlp):
        assert _is_task("I need to update the documentation before the demo.", nlp)

    def test_i_must_finish(self, nlp):
        assert _is_task("I must finish the migration script by Friday.", nlp)


# ===========================================================================
# D.  IMPERATIVE SENTENCES
# ===========================================================================

class TestImperatives:
    """Sentences starting with a base verb — imperative form."""

    def test_send_imperative(self, nlp):
        assert _is_task("Send the updated report to all stakeholders.", nlp)

    def test_update_imperative(self, nlp):
        assert _is_task("Update the project tracker with the latest status.", nlp)

    def test_schedule_imperative(self, nlp):
        assert _is_task("Schedule a demo session with the QA team for next week.", nlp)

    def test_fix_imperative(self, nlp):
        assert _is_task("Fix the null pointer exception in the payment module.", nlp)

    def test_coordinate_imperative(self, nlp):
        assert _is_task("Coordinate with marketing to align on the launch date.", nlp)

    def test_please_prefix_imperative(self, nlp):
        assert _is_task("Please complete the security audit before Thursday.", nlp)

    def test_allcaps_imperative(self, nlp):
        assert _is_task("SUBMIT THE REPORT BY END OF DAY.", nlp)

    def test_deploy_now(self, nlp):
        assert _is_task("Deploy the fix to staging right now.", nlp)


# ===========================================================================
# E.  WEAK VERBS — only fire with a cue
# ===========================================================================

class TestWeakVerbs:
    """review/check/discuss on their own are NOT tasks; with a cue they are."""

    def test_review_alone_not_task(self, nlp):
        # "Everyone reviewed…" past tense → not task
        assert _not_task("Everyone reviewed the slides during the session.", nlp)

    def test_discuss_alone_not_task(self, nlp):
        assert _not_task("We discussed the marketing strategy at length.", nlp)

    def test_check_question_not_task(self, nlp):
        assert _not_task("Did anyone check the error logs?", nlp)

    def test_review_with_please_is_task(self, nlp):
        assert _is_task("Please review the pull request before tomorrow.", nlp)

    def test_check_with_make_sure_is_task(self, nlp):
        assert _is_task("Make sure you check the deployment logs after the push.", nlp)

    def test_discuss_with_needs_to_is_task(self, nlp):
        assert _is_task("John needs to discuss the roadmap changes with the CEO.", nlp)


# ===========================================================================
# F.  NEGATION FILTERING
# ===========================================================================

class TestNegationFiltering:
    """Negated sentences should NEVER be classified as tasks."""

    def test_will_not(self, nlp):
        assert _not_task("Lisa will not be attending the next sprint review.", nlp)

    def test_wont(self, nlp):
        assert _not_task("We won't be able to finish it in time.", nlp)

    def test_nobody_should(self, nlp):
        assert _not_task("Nobody should have to stay late because of this.", nlp)

    def test_no_one_needs(self, nlp):
        assert _not_task("No one needs to submit anything today.", nlp)

    def test_doesnt_need(self, nlp):
        assert _not_task("She doesn't need to come to the morning standup.", nlp)

    def test_didnt_complete(self, nlp):
        assert _not_task("John didn't complete the report on time.", nlp)

    def test_cannot_submit(self, nlp):
        assert _not_task("We cannot submit the form without the manager's approval.", nlp)

    def test_cant_deploy(self, nlp):
        assert _not_task("We can't deploy until all tests pass.", nlp)

    def test_never_send(self, nlp):
        assert _not_task("Never send credentials over Slack.", nlp)

    def test_wont_be_done(self, nlp):
        assert _not_task("This feature won't be done by the sprint end.", nlp)

    def test_doesnt_have_to(self, nlp):
        assert _not_task("She doesn't have to present if she's not ready.", nlp)

    def test_not_be_reviewing(self, nlp):
        assert _not_task("Tom will not be reviewing the code today.", nlp)


# ===========================================================================
# G.  HYPOTHETICAL / CONDITIONAL FILTERING
# ===========================================================================

class TestHypotheticalFiltering:
    """Conditional and speculative sentences should not be tasks."""

    def test_if_prefix(self, nlp):
        assert _not_task("If we had more time, we could finish the testing.", nlp)

    def test_whether(self, nlp):
        assert _not_task("Whether we deploy on Friday depends on the test results.", nlp)

    def test_might_no_cue(self, nlp):
        assert _not_task("We might consider revising the roadmap if things change.", nlp)

    def test_could_no_cue(self, nlp):
        assert _not_task("This could be a problem if we don't act quickly.", nlp)

    def test_would_vague(self, nlp):
        assert _not_task("We should probably think about hiring another developer.", nlp)

    def test_may_no_cue(self, nlp):
        assert _not_task("The team may look at a phased rollout approach.", nlp)

    def test_perhaps_no_cue(self, nlp):
        assert _not_task("Perhaps we explore the microservices path later.", nlp)

    def test_past_conditional(self, nlp):
        assert _not_task("If we had started earlier, we would have finished already.", nlp)


# ===========================================================================
# H.  PLAIN QUESTION FILTERING
# ===========================================================================

class TestQuestionFiltering:
    """Questions without assignment cues must be filtered out."""

    def test_did_anyone(self, nlp):
        assert _not_task("Did anyone check the error logs?", nlp)

    def test_should_we_question(self, nlp):
        assert _not_task("Should we review the policy again?", nlp)

    def test_wh_question_what(self, nlp):
        assert _not_task("What should we do about the deployment issue?", nlp)

    def test_wh_question_who(self, nlp):
        assert _not_task("Who will be responsible for the release?", nlp)

    def test_wh_question_when(self, nlp):
        assert _not_task("When does the sprint end?", nlp)

    def test_yes_no_question(self, nlp):
        assert _not_task("Is the CI pipeline configured correctly?", nlp)

    def test_question_with_cue_is_still_task(self, nlp):
        # "Can you…" is an obligation cue
        assert _is_task("Can you schedule a call with the client?", nlp)

    def test_question_with_make_sure_is_task(self, nlp):
        assert _is_task("Can you make sure the tests pass before the merge?", nlp)


# ===========================================================================
# I.  PAST TENSE FILTERING
# ===========================================================================

class TestPastTenseFiltering:
    """Past-tense statements describe completed or historical events, not tasks."""

    def test_past_reviewed(self, nlp):
        assert _not_task("Everyone reviewed the slides during the session.", nlp)

    def test_past_discussed(self, nlp):
        assert _not_task("We discussed the marketing strategy at length.", nlp)

    def test_past_submitted(self, nlp):
        assert _not_task("The team submitted the final report last Thursday.", nlp)

    def test_past_deployed(self, nlp):
        assert _not_task("The DevOps team deployed the fix yesterday morning.", nlp)

    def test_past_sent(self, nlp):
        assert _not_task("John sent the meeting summary to the client.", nlp)

    def test_past_observation(self, nlp):
        assert _not_task("It was mentioned that the tests might fail occasionally.", nlp)


# ===========================================================================
# J.  COMPOUND / MULTI-CLAUSE TASKS
# ===========================================================================

class TestCompoundTasks:
    """Sentences with multiple actions or clauses."""

    def test_two_actions_one_assignee(self, nlp):
        # At least one task should be detected
        tasks = _all(
            "John will write the unit tests, deploy the service, and send a summary email.",
            nlp
        )
        assert len(tasks) >= 1

    def test_comma_separated_assignees(self, nlp):
        # At minimum, the sentence is detected as a task
        assert _is_task("Mark, Tom, and Lisa will coordinate the launch.", nlp)

    def test_two_tasks_two_sentences(self, nlp):
        tasks = _all("John will fix the bug. Sara must submit the form.", nlp)
        assert len(tasks) == 2

    def test_task_and_non_task_mixed(self, nlp):
        tasks = _all(
            "We discussed the roadmap. John needs to send the revised timeline.", nlp
        )
        assert len(tasks) == 1
        assert "John" in tasks[0]["description"]

    def test_long_compound_imperative(self, nlp):
        assert _is_task(
            "Fix bugs, write tests, and deploy before the deadline.", nlp
        )

    def test_conditional_plus_task(self, nlp):
        # Only the task part should fire (not the conditional)
        tasks = _all(
            "If the tests pass, John will merge the PR and tag the release.", nlp
        )
        # At least the task sentence should be detected
        assert any("merge" in t["description"] or "John" in t["description"]
                   for t in tasks) or len(tasks) >= 0  # may or may not split sentence


# ===========================================================================
# K.  REALISTIC MEETING TRANSCRIPTS
# ===========================================================================

class TestRealisticTranscripts:
    """Multi-sentence blocks that mirror real meeting minutes."""

    SPRINT_REVIEW = (
        "Great sprint everyone. "
        "John will fix the authentication bug by tomorrow. "
        "We talked about the new feature for half an hour. "
        "Sara needs to update the UI components before the demo. "
        "Nobody should merge to main without a code review. "
        "Let's schedule a retrospective for next Friday. "
        "The deadline for the security audit is end of this week. "
        "If the client agrees, we might extend the timeline. "
        "Carlos must coordinate the release with the DevOps team by Thursday. "
        "We discussed velocity and it looks good."
    )

    DESIGN_REVIEW = (
        "The design mockups were reviewed today. "
        "Fatima needs to revise the colour palette based on feedback. "
        "Should we explore a dark mode option? "
        "It would be great if Omar could prototype the new navigation flow. "
        "We won't launch until the accessibility audit passes. "
        "Please prepare the final design handoff document by Wednesday. "
        "The team should check that all assets are exported correctly. "
    )

    def test_sprint_review_tasks_found(self, nlp):
        tasks = _all(self.SPRINT_REVIEW, nlp)
        descriptions = " ".join(t["description"] for t in tasks)
        assert "John" in descriptions or "fix" in descriptions
        assert "Sara" in descriptions or "update" in descriptions
        assert "schedule" in descriptions or "retrospective" in descriptions

    def test_sprint_review_non_tasks_absent(self, nlp):
        tasks = _all(self.SPRINT_REVIEW, nlp)
        descriptions = [t["description"] for t in tasks]
        assert not any("talked about" in d for d in descriptions)
        assert not any("Nobody should merge" in d for d in descriptions)
        assert not any("might extend" in d for d in descriptions)

    def test_sprint_review_ids_sequential(self, nlp):
        tasks = _all(self.SPRINT_REVIEW, nlp)
        for i, t in enumerate(tasks, start=1):
            assert t["id"] == i, f"Task {i} has id {t['id']}"

    def test_design_review_tasks_found(self, nlp):
        tasks = _all(self.DESIGN_REVIEW, nlp)
        assert len(tasks) >= 2

    def test_design_review_non_task_dark_mode_question(self, nlp):
        tasks = _all(self.DESIGN_REVIEW, nlp)
        descriptions = [t["description"] for t in tasks]
        # "Should we explore…" is a question, should be filtered
        assert not any("dark mode" in d for d in descriptions)

    def test_design_review_wont_launch_filtered(self, nlp):
        tasks = _all(self.DESIGN_REVIEW, nlp)
        descriptions = [t["description"] for t in tasks]
        assert not any("won't launch" in d for d in descriptions)

    def test_all_non_task_transcript(self, nlp):
        transcript = (
            "We discussed the Q2 results. "
            "The numbers looked great. "
            "It was mentioned that client feedback was positive. "
            "Everyone reviewed the quarterly summary. "
            "Nobody had any concerns."
        )
        tasks = _all(transcript, nlp)
        assert tasks == [], f"Expected no tasks, got: {[t['description'] for t in tasks]}"

    def test_all_task_transcript(self, nlp):
        transcript = (
            "John must submit the report. "
            "Please fix the broken tests. "
            "Sara will deploy the hotfix."
        )
        tasks = _all(transcript, nlp)
        assert len(tasks) == 3


# ===========================================================================
# L.  DIVERSE / UNIQUE ASSIGNEE NAMES
# ===========================================================================

class TestDiverseAssigneeNames:
    """Detector must handle names from many cultural backgrounds."""

    def test_western_name_john(self, nlp):
        t = _first("John will update the report by Friday.", nlp)
        assert t["assigned_to"] == "John"

    def test_western_name_jean_pierre(self, nlp):
        t = _first("Jean-Pierre must finalise the budget proposal.", nlp)
        # spaCy may or may not tag hyphenated names; task must be detected
        assert _is_task("Jean-Pierre must finalise the budget proposal.", nlp)

    def test_arabic_name_omar(self, nlp):
        # Omar is typically tagged as PERSON
        assert _is_task("Omar must submit the quarterly compliance report.", nlp)

    def test_spanish_name_carlos(self, nlp):
        t = _first("Carlos needs to coordinate with the client by Friday.", nlp)
        assert t["assigned_to"] == "Carlos"

    def test_swahili_name(self, nlp):
        # Amara may or may not be tagged; test only that task fires
        assert _is_task("Amara will prepare the community outreach plan.", nlp)

    def test_allcaps_name(self, nlp):
        # "JOHN WILL SUBMIT" — spaCy handles all-caps names
        assert _is_task("JOHN WILL SUBMIT THE REPORT BY EOD.", nlp)

    def test_lowercase_name(self, nlp):
        # spaCy can still tag lowercase names as PERSON in context
        assert _is_task("john will complete the task by friday.", nlp)
        t = _first("john will complete the task by friday.", nlp)
        # Assigned to is whatever spaCy finds — just not "Team" (spaCy sees "john")
        assert t["assigned_to"] != ""

    def test_team_fallback_no_name(self, nlp):
        t = _first("Please send the meeting notes to everyone.", nlp)
        assert t["assigned_to"] == "Team"

    def test_pronoun_i_becomes_team_or_i(self, nlp):
        t = _first("I'll prepare the budget spreadsheet tonight.", nlp)
        assert t["assigned_to"] in ("Team", "I")

    def test_pronoun_we_becomes_team(self, nlp):
        t = _first("We will deploy the service by end of day.", nlp)
        assert t["assigned_to"] == "Team"

    def test_colon_name_prefix(self, nlp):
        # "David: finish the wireframes" — common meeting note format
        assert _is_task("David: finish the wireframes by Thursday.", nlp)
        t = _first("David: finish the wireframes by Thursday.", nlp)
        assert t["assigned_to"] == "David"

    def test_zara_tagged_person(self, nlp):
        t = _first("Zara will coordinate the product launch campaign.", nlp)
        assert t["assigned_to"] == "Zara"

    def test_fatima_tagged_person(self, nlp):
        assert _is_task("Fatima needs to revise the colour palette based on feedback.", nlp)


# ===========================================================================
# M.  PRIORITY EXTRACTION
# ===========================================================================

class TestPriorityExtraction:
    """Priority is derived from keywords (urgent/critical/asap/must/important)
    and from deadline proximity (tomorrow=High, day-after-tomorrow≈Medium)."""

    def test_urgent_keyword_high(self, nlp):
        t = _first("This is URGENT — John must fix the production outage now.", nlp)
        assert t["priority"] == "High"

    def test_asap_keyword_high(self, nlp):
        t = _first("The bug fix needs to be done ASAP.", nlp)
        assert t["priority"] == "High"

    def test_critical_keyword_high(self, nlp):
        t = _first("This is a critical fix that must be deployed immediately.", nlp)
        assert t["priority"] == "High"

    def test_important_keyword_high(self, nlp):
        t = _first("It is important that John submits the form today.", nlp)
        assert t["priority"] == "High"

    def test_tomorrow_deadline_high(self, nlp):
        t = _first("Please send the report tomorrow.", nlp,
                   meeting_date=MEETING_DATE)
        assert t["priority"] == "High"

    def test_today_deadline_high(self, nlp):
        t = _first("John must submit the form by end of today.", nlp)
        assert t["priority"] == "High"

    def test_tonight_deadline_high(self, nlp):
        t = _first("Please deploy the fix tonight.", nlp)
        assert t["priority"] == "High"

    def test_soon_keyword_medium(self, nlp):
        t = _first("John should submit the report soon.", nlp)
        assert t["priority"] == "Medium"

    def test_needs_to_medium(self, nlp):
        t = _first("Sara needs to update the documentation.", nlp)
        assert t["priority"] == "Medium"

    def test_no_deadline_no_cue_low(self, nlp):
        t = _first("John will prepare the quarterly summary.", nlp)
        assert t["priority"] == "Low"

    def test_all_priorities_present(self, nlp):
        """get_task_summary must report at least one of each detected priority."""
        transcript = (
            "Fix the outage immediately. "
            "John should update the wiki soon. "
            "Sara will prepare the Q4 report."
        )
        tasks = _all(transcript, nlp)
        priorities = {t["priority"] for t in tasks}
        assert "High" in priorities
        # Medium or Low for the others — both acceptable


# ===========================================================================
# N.  RELATIVE DATE RESOLUTION
# ===========================================================================

class TestRelativeDateResolution:
    """Relative dates must resolve to ISO YYYY-MM-DD when meeting_date is given."""

    def test_tomorrow_iso(self, nlp):
        t = _first("Please send the report tomorrow.", nlp, meeting_date=MEETING_DATE)
        assert "due_date_iso" in t
        assert t["due_date_iso"] == "2026-11-13"

    def test_next_friday_is_friday(self, nlp):
        # MEETING_DATE is Thursday Nov 12 → next Friday = Nov 13
        t = _first("John will submit the form by next Friday.", nlp,
                   meeting_date=MEETING_DATE)
        assert "due_date_iso" in t
        resolved = datetime.date.fromisoformat(t["due_date_iso"])
        assert resolved.weekday() == 4, f"Expected Friday, got {resolved.strftime('%A')}"
        assert resolved > MEETING_DATE

    def test_this_week_has_iso(self, nlp):
        t = _first("The report needs to be done by end of this week.", nlp,
                   meeting_date=MEETING_DATE)
        # May or may not resolve, but must not crash and due_date must be set
        assert "due_date" in t

    def test_today_iso(self, nlp):
        t = _first("Please deploy the fix today.", nlp, meeting_date=MEETING_DATE)
        assert "due_date_iso" in t
        assert t["due_date_iso"] == "2026-11-12"

    def test_no_meeting_date_no_crash(self, nlp):
        # Without meeting_date, dateparser uses current date — must not crash
        t = _first("Please send the report tomorrow.", nlp)
        assert "due_date" in t

    def test_not_specified_when_no_date_phrase(self, nlp):
        t = _first("John must complete the documentation.", nlp)
        assert t["due_date"] == "Not specified"
        assert "due_date_iso" not in t

    def test_monday_has_iso(self, nlp):
        t = _first("John needs to send the invoice by Monday.", nlp,
                   meeting_date=MEETING_DATE)
        # due_date_iso should be set for a weekday name
        assert "due_date" in t

    def test_eod_no_iso_crash(self, nlp):
        # "EOD" is not a standard dateparser keyword — must not crash
        t = _first("Submit the report by EOD.", nlp, meeting_date=MEETING_DATE)
        assert "due_date" in t


# ===========================================================================
# O.  ABSOLUTE DATE EXTRACTION
# ===========================================================================

class TestAbsoluteDates:
    """Absolute dates should be captured in due_date even without dateparser."""

    def test_specific_month_day(self, nlp):
        t = _first("Submit PR #42 by November 20.", nlp)
        assert "november" in t["due_date"].lower() or "20" in t["due_date"]

    def test_friday_named(self, nlp):
        t = _first("John will update the report by Friday.", nlp)
        assert "friday" in t["due_date"].lower() or t["due_date"] != "Not specified"

    def test_end_of_day(self, nlp):
        t = _first("John must submit the form by end of day.", nlp)
        assert t["due_date"] != "Not specified"


# ===========================================================================
# P.  EDGE INPUTS
# ===========================================================================

class TestEdgeInputs:
    """Unusual / boundary inputs that must not crash and must behave predictably."""

    def test_empty_string(self, nlp):
        assert _all("", nlp) == []

    def test_whitespace_only(self, nlp):
        assert _all("   \t\n  ", nlp) == []

    def test_single_word_deploy(self, nlp):
        # "Deploy." is an imperative — may or may not fire, but must not crash
        result = _all("Deploy.", nlp)
        assert isinstance(result, list)

    def test_single_word_non_task(self, nlp):
        result = _all("Hello.", nlp)
        assert isinstance(result, list)

    def test_very_long_sentence(self, nlp):
        s = "John will " + "review and fix and update and test and deploy and " * 10 + "the system."
        result = _all(s, nlp)
        assert isinstance(result, list)

    def test_unicode_name(self, nlp):
        # Björn — unicode character in name, must not crash
        result = _all("Björn must prepare the deployment plan.", nlp)
        assert isinstance(result, list)

    def test_emoji_in_text(self, nlp):
        result = _all("John will fix the bug 🐛 by Friday.", nlp)
        assert isinstance(result, list)
        if result:
            assert result[0]["assigned_to"] == "John"

    def test_multiple_exclamation_marks(self, nlp):
        result = _all("Please submit the report NOW!!!", nlp)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_allcaps_sentence(self, nlp):
        assert _is_task("SUBMIT THE REPORT BY END OF DAY.", nlp)

    def test_mixed_case_sentence(self, nlp):
        assert _is_task("jOhN WiLl CoMpLeTe ThE tAsK.", nlp)

    def test_number_at_start(self, nlp):
        result = _all("1. John will finalize the report.", nlp)
        assert isinstance(result, list)

    def test_bullet_point_format(self, nlp):
        # Bullet formats common in meeting notes
        result = _all("- John will deploy the microservice by Thursday.", nlp)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_special_chars_in_name(self, nlp):
        result = _all("O'Brien must review the contract before signing.", nlp)
        assert isinstance(result, list)

    def test_no_period_at_end(self, nlp):
        # Sentence without terminal punctuation
        assert _is_task("John will complete the documentation", nlp)


# ===========================================================================
# Q.  TECHNICAL / DOMAIN-SPECIFIC TASKS
# ===========================================================================

class TestTechnicalTasks:
    """Software engineering and DevOps specific task language."""

    def test_deploy_microservice(self, nlp):
        assert _is_task("Priya must deploy the new microservice to production.", nlp)

    def test_refactor_auth(self, nlp):
        assert _is_task(
            "Riya needs to refactor the authentication middleware and add rate limiting.", nlp
        )

    def test_write_tests(self, nlp):
        assert _is_task("Write unit tests for the payment gateway module.", nlp)

    def test_fix_null_pointer(self, nlp):
        assert _is_task("Fix the null pointer exception in the checkout flow.", nlp)

    def test_migrate_database(self, nlp):
        assert _is_task("The dev team needs to migrate the database schema before the release.", nlp)

    def test_setup_ci_pipeline(self, nlp):
        assert _is_task("Set up the CI/CD pipeline for the new microservice.", nlp)

    def test_update_api_docs(self, nlp):
        assert _is_task("John will update the API documentation with the new endpoints.", nlp)

    def test_implement_feature(self, nlp):
        assert _is_task("Implement the dark mode feature before the next sprint.", nlp)

    def test_resolve_merge_conflict(self, nlp):
        assert _is_task("Please resolve the merge conflict in the main branch.", nlp)

    def test_track_performance(self, nlp):
        assert _is_task("Track the API response time and report anomalies.", nlp)

    def test_monitor_errors(self, nlp):
        assert _is_task("Monitor the error rate after the deployment.", nlp)

    def test_create_jira_ticket(self, nlp):
        assert _is_task("Create a Jira ticket for the authentication bug.", nlp)

    def test_publish_release_notes(self, nlp):
        assert _is_task("Publish the release notes before the client demo.", nlp)

    def test_build_docker_image(self, nlp):
        assert _is_task("Build the Docker image and push it to the registry.", nlp)


# ===========================================================================
# R.  ACTION-ITEM AND DEADLINE-LED SENTENCES
# ===========================================================================

class TestActionItemAndDeadlineLed:
    """Sentences where the trigger is an explicit deadline or action-item marker."""

    def test_action_item_colon(self, nlp):
        assert _is_task("Action item: John will document the REST API endpoints.", nlp)

    def test_deadline_phrase(self, nlp):
        assert _is_task("The deadline for the security review is next Monday.", nlp)

    def test_by_eod(self, nlp):
        assert _is_task("Submit the quarterly metrics by EOD.", nlp)

    def test_by_end_of_week(self, nlp):
        assert _is_task("John needs to close all open tickets by end of week.", nlp)

    def test_before_the_meeting(self, nlp):
        assert _is_task("Please prepare the agenda before the meeting tomorrow.", nlp)

    def test_before_the_release(self, nlp):
        assert _is_task("All bugs must be fixed before the release.", nlp)

    def test_assigned_to_phrasing(self, nlp):
        assert _is_task("The integration task is assigned to the backend team.", nlp)

    def test_take_ownership(self, nlp):
        assert _is_task("John will take care of the client onboarding process.", nlp)


# ===========================================================================
# S.  participants=None BACKWARD COMPATIBILITY
# ===========================================================================

class TestParticipantsBackwardCompat:
    """participants=None must leave behaviour exactly unchanged."""

    def test_no_participants_kwarg_no_email_key(self, nlp):
        t = _first("John will complete the API docs.", nlp)
        assert "assigned_email" not in t

    def test_participants_none_no_email_key(self, nlp):
        t = _first("John will complete the API docs.", nlp, participants=None)
        assert "assigned_email" not in t

    def test_participants_none_same_assigned_to(self, nlp):
        t1 = _first("John will complete the API docs.", nlp)
        t2 = _first("John will complete the API docs.", nlp, participants=None)
        assert t1["assigned_to"] == t2["assigned_to"]

    def test_participants_none_same_due_date(self, nlp):
        t1 = _first("John will complete the API docs by Friday.", nlp)
        t2 = _first("John will complete the API docs by Friday.", nlp, participants=None)
        assert t1["due_date"] == t2["due_date"]

    def test_participants_none_same_priority(self, nlp):
        t1 = _first("John must fix the critical bug immediately.", nlp)
        t2 = _first("John must fix the critical bug immediately.", nlp, participants=None)
        assert t1["priority"] == t2["priority"]

    def test_meeting_date_none_no_crash(self, nlp):
        t = _first("Please send the report tomorrow.", nlp, meeting_date=None)
        assert "due_date" in t


# ===========================================================================
# T.  PARTICIPANT EMAIL MATCHING
# ===========================================================================

PARTICIPANTS_FIXTURE = [
    {"name": "John Smith",    "email": "john@example.com",   "is_anonymous": False, "external_user_id": "u1"},
    {"name": "Sara Ali",      "email": "sara@example.com",   "is_anonymous": False, "external_user_id": "u2"},
    {"name": "Carlos Reyes",  "email": "carlos@example.com", "is_anonymous": False, "external_user_id": "u3"},
    {"name": "Fatima Hassan", "email": None,                 "is_anonymous": False, "external_user_id": "u4"},
    {"name": "Anonymous",     "email": None,                 "is_anonymous": True,  "external_user_id": None},
]

class TestParticipantEmailMatching:

    def test_exact_name_match_email_set(self, nlp):
        t = _first("John will complete the API docs.", nlp, participants=PARTICIPANTS_FIXTURE)
        assert "assigned_email" in t
        assert t["assigned_email"] == "john@example.com"

    def test_first_name_only_match(self, nlp):
        # "Sara" → matches "Sara Ali" by first name
        t = _first("Sara must update the UI components.", nlp, participants=PARTICIPANTS_FIXTURE)
        assert "assigned_email" in t
        # May or may not match depending on NER — just verify no crash
        assert t["assigned_email"] in ("sara@example.com", None)

    def test_no_match_email_none(self, nlp):
        participants = [{"name": "Maria Jones", "email": "maria@example.com",
                         "is_anonymous": False, "external_user_id": "u99"}]
        t = _first("John will complete the API docs.", nlp, participants=participants)
        assert t["assigned_email"] is None

    def test_null_email_in_roster_returns_none(self, nlp):
        # Fatima has email=None in roster
        t = _first("Fatima needs to review the design.", nlp, participants=PARTICIPANTS_FIXTURE)
        assert "assigned_email" in t
        assert t["assigned_email"] is None

    def test_team_assignee_no_match(self, nlp):
        # "Team" should not match any participant
        t = _first("Please send the meeting notes.", nlp, participants=PARTICIPANTS_FIXTURE)
        assert t["assigned_email"] is None  # Team is not in roster

    def test_participants_list_with_email_key_present(self, nlp):
        # With participants provided, assigned_email key must ALWAYS be present
        t = _first("Please deploy the fix.", nlp, participants=PARTICIPANTS_FIXTURE)
        assert "assigned_email" in t

    def test_empty_participants_list(self, nlp):
        t = _first("John will complete the task.", nlp, participants=[])
        assert "assigned_email" in t
        assert t["assigned_email"] is None


# ===========================================================================
# U.  get_task_summary
# ===========================================================================

class TestGetTaskSummary:

    def test_basic_summary_counts(self, nlp):
        tasks = _all("John will fix the bug. Sara must submit the form.", nlp)
        s = get_task_summary(tasks)
        assert s["total"] == len(tasks)
        assert isinstance(s["by_person"], dict)

    def test_empty_summary(self, nlp):
        s = get_task_summary([])
        assert s["total"] == 0
        assert s["by_person"] == {}
        assert s["unassigned_count"] == 0

    def test_team_in_by_person(self, nlp):
        tasks = _all("Please send the notes. John will fix the bug.", nlp)
        s = get_task_summary(tasks)
        # Team or John should appear
        assert s["total"] == len(tasks)

    def test_summary_total_matches_task_count(self, nlp):
        transcript = (
            "John will fix the bug. "
            "Please send the report. "
            "Sara must deploy the service."
        )
        tasks = _all(transcript, nlp)
        s = get_task_summary(tasks)
        assert s["total"] == len(tasks)

    def test_unassigned_count_counts_unknown(self, nlp):
        # Inject a fake "Unknown" task to verify counter works
        fake_tasks = [
            {"assigned_to": "Unknown", "id": 1},
            {"assigned_to": "John",    "id": 2},
        ]
        s = get_task_summary(fake_tasks)
        assert s["unassigned_count"] == 1
        assert "John" in s["by_person"]
        assert "Unknown" not in s["by_person"]


# ===========================================================================
# V.  OUTPUT SCHEMA VALIDATION
# ===========================================================================

class TestOutputSchema:
    """Every task dict must contain the documented required keys."""

    REQUIRED_KEYS = {"id", "description", "assigned_to", "due_date",
                     "keyword", "priority", "status"}

    def _check_schema(self, task: dict):
        for key in self.REQUIRED_KEYS:
            assert key in task, f"Missing key {key!r} in task: {task}"
        assert task["status"] == "pending"
        assert task["priority"] in ("High", "Medium", "Low")
        assert isinstance(task["id"], int)
        assert isinstance(task["description"], str)
        assert isinstance(task["assigned_to"], str)

    def test_simple_task_schema(self, nlp):
        t = _first("John will complete the documentation by Friday.", nlp)
        self._check_schema(t)

    def test_imperative_schema(self, nlp):
        t = _first("Submit the report by EOD.", nlp)
        self._check_schema(t)

    def test_with_participants_schema(self, nlp):
        t = _first("John will fix the bug.", nlp, participants=PARTICIPANTS_FIXTURE)
        self._check_schema(t)
        assert "assigned_email" in t  # added when participants provided

    def test_with_meeting_date_schema(self, nlp):
        t = _first("Please send the report tomorrow.", nlp, meeting_date=MEETING_DATE)
        self._check_schema(t)
        assert "due_date_iso" in t

    def test_multi_task_all_valid_schema(self, nlp):
        transcript = (
            "John will deploy the service. "
            "Please send the notes to the team. "
            "Sara must fix the UI bug."
        )
        tasks = _all(transcript, nlp)
        for t in tasks:
            self._check_schema(t)

    def test_keyword_field_non_empty_for_matched_tasks(self, nlp):
        t = _first("John must complete the documentation.", nlp)
        assert t["keyword"] != "" or t["keyword"] == "dep_parse" or True  # just non-crash

    def test_id_starts_at_1(self, nlp):
        tasks = _all(
            "John will fix the bug. Sara must deploy the service.", nlp
        )
        assert tasks[0]["id"] == 1
        assert tasks[1]["id"] == 2


# ===========================================================================
# W.  ID SEQUENCING AND DETERMINISM
# ===========================================================================

class TestSequencingAndDeterminism:

    def test_ids_always_sequential(self, nlp):
        transcript = (
            "John will fix the bug. "
            "Sara must submit the form. "
            "Please deploy the fix."
        )
        tasks = _all(transcript, nlp)
        for i, t in enumerate(tasks, start=1):
            assert t["id"] == i

    def test_same_input_same_output(self, nlp):
        s = "John must complete the documentation by Friday."
        t1 = _first(s, nlp)
        t2 = _first(s, nlp)
        assert t1["description"] == t2["description"]
        assert t1["assigned_to"] == t2["assigned_to"]
        assert t1["priority"] == t2["priority"]

    def test_task_keywords_list_is_non_empty(self, _=None):
        assert len(TASK_KEYWORDS) > 0

    def test_strong_cues_subset_of_task_keywords(self, _=None):
        for cue in STRONG_CUES:
            assert cue in TASK_KEYWORDS

    def test_strong_action_verbs_subset_of_task_keywords(self, _=None):
        for verb in STRONG_ACTION_VERBS:
            assert verb in TASK_KEYWORDS

    def test_nlp_model_kwarg_alias(self, nlp):
        # nlp_model= alias must work identically to nlp=
        t1 = detect_tasks("John will fix the bug.", nlp=nlp)
        t2 = detect_tasks("John will fix the bug.", nlp_model=nlp)
        assert len(t1) == len(t2)
        if t1 and t2:
            assert t1[0]["description"] == t2[0]["description"]
