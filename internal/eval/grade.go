package eval

import (
	"strings"

	"github.com/richardknop/my-agent/internal/agent"
	"github.com/richardknop/my-agent/internal/data"
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

// GradeReport compares a report with the expected answer. deploys is the
// scenario's deploy history, used to tell which service a free-text
// triggering_change refers to. A nil report (the run failed) gets a zero
// Grade.
func GradeReport(exp Expect, deploys []data.Deploy, r *agent.Report) Grade {
	if r == nil {
		return Grade{}
	}
	var g Grade
	g.RootCauseService = strings.EqualFold(strings.TrimSpace(r.RootCauseService), exp.RootCauseService)
	g.TriggeringChange = gradeChange(exp.TriggeringChange, deploys, r)

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
// "v2.14.0", "payments v2.14.0", "v2.14.0 deploy to the payment service",
// "v2.14.0 deployment reducing payments-db connection pool timeout". It must
// contain the expected version, and refer to the expected service:
//
//   - If it names services that deploy (any service in the deploy history),
//     the expected one must be among them, so "checkout v2.14.0" fails.
//     Services that never deploy, like a database the change configured,
//     are context, not attribution.
//   - If it names none, the version stands for the deploy when exactly one
//     service shipped it. Otherwise (no such deploy, or the version number
//     is shared) the report's root_cause_service stands in for the service.
func gradeChange(exp *Change, deploys []data.Deploy, r *agent.Report) bool {
	text := strings.ToLower(strings.TrimSpace(r.TriggeringChange))
	if exp == nil {
		return text == ""
	}
	version := strings.ToLower(exp.Version)
	if !strings.Contains(text, version) {
		return false
	}

	deployed := map[string]bool{} // services that deploy
	shipped := map[string]bool{}  // services that shipped the expected version
	for _, d := range deploys {
		svc := strings.ToLower(d.Service)
		deployed[svc] = true
		if strings.ToLower(d.Version) == version {
			shipped[svc] = true
		}
	}

	if named := namedServices(text, deployed); len(named) > 0 {
		return named[strings.ToLower(exp.Service)]
	}
	if len(shipped) == 1 {
		return shipped[strings.ToLower(exp.Service)]
	}
	return strings.EqualFold(strings.TrimSpace(r.RootCauseService), exp.Service)
}

// namedServices returns the services mentioned in text, matched as whole
// words and ignoring a trailing "s", so "payment service" names payments but
// "payments-db" names only payments-db.
func namedServices(text string, services map[string]bool) map[string]bool {
	words := map[string]bool{}
	for _, w := range strings.FieldsFunc(text, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-')
	}) {
		words[strings.TrimSuffix(w, "s")] = true
	}
	named := map[string]bool{}
	for s := range services {
		if words[strings.TrimSuffix(s, "s")] {
			named[s] = true
		}
	}
	return named
}
