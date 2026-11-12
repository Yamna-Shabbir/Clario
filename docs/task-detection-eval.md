
---
## Eval run 2026-11-12 15:16 *(BASELINE)*

### dev (20) split  (P=0.714, R=0.909, F1=0.800)

| TP | FP | FN | TN |
|----|----|----|----|
| 10 | 4 | 1 | 5 |

**False Positives:**
- [14] *Nobody should have to stay late because of this.* — negation with should
- [16] *Should we review the policy again?* — question form of should, not assignment
- [18] *Lisa will not be attending the next sprint review.* — negation with will
- [20] *If we had more time, we could finish the testing.* — hypothetical past conditional

**False Negatives:**
- [10] *Ahmed is responsible for the backend deployment this week.* — responsible for + deadline

### held-out (10) split  (P=0.714, R=0.833, F1=0.769)

| TP | FP | FN | TN |
|----|----|----|----|
| 5 | 2 | 1 | 2 |

**False Positives:**
- [24] *Is there anything we should check before going live?* — question, no specific assignee or action
- [26] *We should probably think about hiring another developer.* — vague deliberation, no assignment

**False Negatives:**
- [30] *Can you make sure the CI pipeline passes before merging?* — can you + ensure + action


---
## Eval run 2026-11-12 15:17

### dev (20) split  (P=0.909, R=0.909, F1=0.909)

| TP | FP | FN | TN |
|----|----|----|----|
| 10 | 1 | 1 | 8 |

**False Positives:**
- [15] *The meeting covered topics like budget and hiring.* — summary sentence, no task

**False Negatives:**
- [6] *It would be great if Sara could handle the slides.* — indirect/polite request

### held-out (10) split  (P=1.000, R=1.000, F1=1.000)

| TP | FP | FN | TN |
|----|----|----|----|
| 6 | 0 | 0 | 4 |

**False Positives:** none

**False Negatives:** none


---
## Eval run 2026-11-12 15:18

### dev (20) split  (P=0.917, R=1.000, F1=0.957)

| TP | FP | FN | TN |
|----|----|----|----|
| 11 | 1 | 0 | 8 |

**False Positives:**
- [15] *The meeting covered topics like budget and hiring.* — summary sentence, no task

**False Negatives:** none

### held-out (10) split  (P=1.000, R=1.000, F1=1.000)

| TP | FP | FN | TN |
|----|----|----|----|
| 6 | 0 | 0 | 4 |

**False Positives:** none

**False Negatives:** none


---
## Eval run 2026-11-12 15:19

### dev (20) split  (P=1.000, R=1.000, F1=1.000)

| TP | FP | FN | TN |
|----|----|----|----|
| 11 | 0 | 0 | 9 |

**False Positives:** none

**False Negatives:** none

### held-out (10) split  (P=1.000, R=1.000, F1=1.000)

| TP | FP | FN | TN |
|----|----|----|----|
| 6 | 0 | 0 | 4 |

**False Positives:** none

**False Negatives:** none

