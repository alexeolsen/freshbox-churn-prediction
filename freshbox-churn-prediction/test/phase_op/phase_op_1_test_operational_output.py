#!/usr/bin/env python
"""
Generate Operational Output Document

Converts technical ML results into actionable business outcomes
for sales, retention, and leadership teams.
"""

from src.shared.phase_0_3_operational_actions import (
    print_executive_brief,
    print_retention_team_playbook,
    print_contact_prioritisation,
    print_feature_to_tactic_mapping,
    print_weekly_operations_guide,
    print_success_metrics,
    print_faq_for_retention_team,
    print_presentation_talking_points
)


if __name__ == "__main__":
    print("\n[*] OPERATIONAL ACTIONS: FROM ML MODEL TO BUSINESS OUTCOMES\n")

    print_executive_brief()
    print_retention_team_playbook()
    print_contact_prioritisation()
    print_feature_to_tactic_mapping()
    print_weekly_operations_guide()
    print_success_metrics()
    print_faq_for_retention_team()
    print_presentation_talking_points()

    print("\n" + "=" * 100)
    print("[SUCCESS] Operational actions document generated!")
    print("=" * 100)
    print("\nUSE THIS FOR:")
    print("  1. Briefing retention team on what to do")
    print("  2. Communicating to leadership in business terms (not technical)")
    print("  3. Setting realistic expectations on revenue impact")
    print("  4. Planning weekly operations and tracking success")
    print("  5. Answering questions from stakeholders\n")
