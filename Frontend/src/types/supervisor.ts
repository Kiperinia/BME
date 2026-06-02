export interface SupervisorIssue {
  type: string
  severity: string
  message: string
  location?: string | null
  evidenceRefs: string[]
}

export interface SupervisorDecision {
  reportId: string
  status: string
  riskLevel: string
  issues: SupervisorIssue[]
  auditId: string
  rationale?: string | null
  hardCase: boolean
  routing: string[]
  metadata: Record<string, unknown>
}
