package eval

import (
	"strings"

	"github.com/richardknop/my-agent/internal/agent"
)

// Grade is how one report scored against a scenario's expected answer.
type Grade struct {
	RootCauseService bool `json:"root_cause_service"`
	TriggeringChange bool `json:"triggering_change"`
	// AffectedRecall is the share of the expected affected services the
	// report named, from 0 to 1.
	AffectedRecall float64 `json:"affected_recall"`
	// WronglyAffected lists healthy control services the report named as
	// affected.
	WronglyAffected []string `json:"wrongly_affected,omitempty"`
	// Pass means the report got the root cause right: the service and the
	// change that triggered it. Affected services are scored but don't
	// decide it, since they measure thoroughness, not correctness.
	Pass bool `json:"pass"`
}

// GradeReport compares a report with the expected answer. A nil report (the
// run failed) gets a zero Grade.
func GradeReport(exp Expect, r *agent.Report) Grade {
	if r == nil {
		return Grade{}
	}
	var g Grade
	g.RootCauseService = strings.EqualFold(strings.TrimSpace(r.RootCauseService), exp.RootCauseService)

	got := strings.ToLower(strings.TrimSpace(r.TriggeringChange))
	if exp.TriggeringChange == "" {
		g.TriggeringChange = got == ""
	} else {
		g.TriggeringChange = strings.Contains(got, strings.ToLower(exp.TriggeringChange))
	}

	named := map[string]bool{}
	for _, s := range r.AffectedServices {
		named[strings.ToLower(strings.TrimSpace(s))] = true
	}
	if len(exp.AffectedServices) == 0 {
		g.AffectedRecall = 1
	} else {
		found := 0
		for _, s := range exp.AffectedServices {
			if named[strings.ToLower(s)] {
				found++
			}
		}
		g.AffectedRecall = float64(found) / float64(len(exp.AffectedServices))
	}
	for _, s := range exp.UnaffectedServices {
		if named[strings.ToLower(s)] {
			g.WronglyAffected = append(g.WronglyAffected, s)
		}
	}

	g.Pass = g.RootCauseService && g.TriggeringChange
	return g
}
