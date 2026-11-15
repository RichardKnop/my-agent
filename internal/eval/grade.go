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

// GradeReport compares a report with the expected answer. services are all
// the service names in the scenario's dataset, used to tell which service a
// free-text triggering_change refers to. A nil report (the run failed) gets
// a zero Grade.
func GradeReport(exp Expect, services []string, r *agent.Report) Grade {
	if r == nil {
		return Grade{}
	}
	var g Grade
	g.RootCauseService = strings.EqualFold(strings.TrimSpace(r.RootCauseService), exp.RootCauseService)
	g.TriggeringChange = gradeChange(exp.TriggeringChange, services, r)

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

// gradeChange checks triggering_change, which models write freely:
// "v2.14.0", "payments v2.14.0", "v2.14.0 deploy to the payment service".
// It must contain the expected version, and refer to the expected service:
// if it names any service, the expected one must be among them; if it names
// none ("v2.14.0"), the report's root_cause_service stands in for it.
func gradeChange(exp *Change, services []string, r *agent.Report) bool {
	text := strings.ToLower(strings.TrimSpace(r.TriggeringChange))
	if exp == nil {
		return text == ""
	}
	if !strings.Contains(text, strings.ToLower(exp.Version)) {
		return false
	}
	named := namedServices(text, services)
	if len(named) == 0 {
		return strings.EqualFold(strings.TrimSpace(r.RootCauseService), exp.Service)
	}
	return named[strings.ToLower(exp.Service)]
}

// namedServices returns the services mentioned in text, matched as whole
// words and ignoring a trailing "s", so "payment service" names payments but
// "payments-db" names only payments-db.
func namedServices(text string, services []string) map[string]bool {
	words := map[string]bool{}
	for _, w := range strings.FieldsFunc(text, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-')
	}) {
		words[strings.TrimSuffix(w, "s")] = true
	}
	named := map[string]bool{}
	for _, s := range services {
		s = strings.ToLower(s)
		if words[strings.TrimSuffix(s, "s")] {
			named[s] = true
		}
	}
	return named
}
