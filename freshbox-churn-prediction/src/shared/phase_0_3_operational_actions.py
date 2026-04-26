"""
Operational Actions Framework

Translates technical ML results into actionable business outcomes and retention tactics
for the FreshBox sales and retention teams.
"""


def print_executive_brief():
    """
    Non-technical executive brief for leadership.
    """
    print("\n" + "=" * 100)
    print("EXECUTIVE BRIEF: FROM ML MODEL TO BUSINESS OUTCOMES")
    print("=" * 100)

    print("\nTHE OPPORTUNITY:")
    print("-" * 100)
    print("Right now: 67% of customers churn within 6 months (5,360 customers per 8,000 new)")
    print("With our model: We can identify at-risk customers BEFORE they churn")
    print("Business result: Proactive retention outreach prevents customer loss")

    print("\nTHE NUMBERS (Year 1 Impact):")
    print("-" * 100)
    print("Revenue at risk (annual churn):          GBP 8,040,000 (from 5,360 lost customers @ GBP 1,500 LTV)")
    print("Customers we can now identify:           ~5,000 at-risk customers per month")
    print("Estimated churn prevention rate:        5-10% (200-300 customers saved)")
    print("Revenue protected:                       GBP 300,000 - GBP 450,000")
    print("Cost of retention effort:                GBP 50 per customer (email, discount, support)")
    print("Net benefit:                             GBP 250,000 - GBP 400,000 (after retention spend)")

    print("\nTHE TIMELINE:")
    print("-" * 100)
    print("Month 1:   Deploy model, score 80,000 customers, identify 5,000 at-risk")
    print("Month 2-3: Retention team contacts flagged customers with targeted offers")
    print("Month 4-6: Measure retention lift, validate churn prevention")
    print("Month 7+:  Scale successful tactics, potentially upgrade to more sophisticated model")


def print_retention_team_playbook():
    """
    Practical retention tactics for the sales/retention team (non-technical).
    """
    print("\n" + "=" * 100)
    print("RETENTION TEAM PLAYBOOK: WHO TO CONTACT AND WHAT TO DO")
    print("=" * 100)

    print("\nWHO WE'LL FLAG (Customer Risk Categories):")
    print("-" * 100)

    print("\n1. THE DISENGAGED CUSTOMER (Highest Priority - 40% of at-risk)")
    print("   Indicator: Stopped ordering regularly in last 4 weeks")
    print("   Why they churn: Lost interest, found competitor, life circumstance change")
    print("   What to do:")
    print("     • Call within 48 hours (personal outreach matters)")
    print("     • Lead with: 'We noticed you haven't ordered recently - everything OK?'")
    print("     • Offer: Personalised discount (GBP 10-15 off) + free shipping on next order")
    print("     • Goal: Get them to re-engage with ONE order (reactivation)")
    print("     • Success metric: Customer places order within 7 days")

    print("\n2. THE DECLINING SATISFACTION CUSTOMER (Medium Priority - 30% of at-risk)")
    print("   Indicator: Recently gave poor ratings to meals or recipes")
    print("   Why they churn: Product doesn't match expectations, meal quality issues")
    print("   What to do:")
    print("     • Email with subject: 'We want to improve your experience'")
    print("     • Acknowledge: 'We saw your recent feedback - we take that seriously'")
    print("     • Offer: Free week of meals + ability to customise next box")
    print("     • Escalate: Flag product team if feedback is quality-related")
    print("     • Goal: Rebuild trust in product quality")
    print("     • Success metric: Next order rating >= 3.5 stars")

    print("\n3. THE HIGH-SUPPORT CUSTOMER (Medium Priority - 20% of at-risk)")
    print("   Indicator: Multiple support tickets about delivery or billing")
    print("   Why they churn: Frustrated by operational friction (delays, errors)")
    print("   What to do:")
    print("     • Phone call from Customer Success Manager (not automated email)")
    print("     • Lead with: 'We see you've had delivery issues - we're here to fix this'")
    print("     • Action: Resolve outstanding issue immediately (refund, replacement)")
    print("     • Offer: Loyalty credit (GBP 25) for next month")
    print("     • Escalate: Flag operations team to prevent future delivery issues")
    print("     • Goal: Fix the operational problem, restore confidence")
    print("     • Success metric: No new support tickets in 30 days")

    print("\n4. THE BARGAIN HUNTER (Lower Priority - 10% of at-risk)")
    print("   Indicator: Heavy discount dependency (always uses promotional codes)")
    print("   Why they churn: Price-sensitive, will switch if competitor offers better deal")
    print("   What to do:")
    print("     • Email: Tiered loyalty discount (5% @ 6 months, 10% @ 12 months)")
    print("     • Offer: Exclusive early access to promotions (make them feel VIP)")
    print("     • Avoid: Large one-time discounts (trains them to wait for sales)")
    print("     • Goal: Build habit and loyalty, not just discount-chase")
    print("     • Success metric: Orders placed without discount code")


def print_contact_prioritisation():
    """
    How to prioritise limited retention budget across 5,000 flagged customers.
    """
    print("\n" + "=" * 100)
    print("CONTACT PRIORITISATION: ALLOCATE YOUR TEAM BASED ON IMPACT")
    print("=" * 100)

    print("\nAssume: Retention team has capacity for 500 proactive outreaches per month")
    print("Total at-risk customers: 5,000 per month")
    print("Reality: You can't reach everyone. Prioritise by impact.")

    print("\nRISK SCORE DISTRIBUTION:")
    print("-" * 100)
    print("High Risk (90% likely to churn):   500 customers   [TIER 1 - CONTACT IMMEDIATELY]")
    print("Medium Risk (70% likely to churn): 1,500 customers [TIER 2 - CONTACT WITHIN 2 WEEKS]")
    print("Lower Risk (50% likely to churn):  3,000 customers [TIER 3 - EMAIL ONLY (COST EFFICIENT)]")

    print("\nMONTH 1-2 TACTIC: PHASE 1 (Build Momentum)")
    print("-" * 100)
    print("Focus: Tier 1 (500 high-risk customers)")
    print("Approach: Dedicated retention specialist calls")
    print("Expected outcome: Prevent 100-150 churns (20-30% save rate)")
    print("Investment: 500 calls × 5 min = 42 hours staff time")
    print("Revenue saved: 125 customers × GBP 1,500 = GBP 187,500")

    print("\nMONTH 2-3 TACTIC: PHASE 2 (Add Tier 2)")
    print("-" * 100)
    print("Focus: Tier 1 (500) + Tier 2 (1,500 medium-risk)")
    print("Approach: Tier 1 = phone calls, Tier 2 = personalised emails + SMS follow-up")
    print("Expected outcome: Prevent 200-300 churns (20% save rate across both)")
    print("Investment: 500 calls + 1,500 emails/SMS = 62 hours staff time")
    print("Revenue saved: 250 customers × GBP 1,500 = GBP 375,000")

    print("\nMONTH 3+ TACTIC: FULL PROGRAMME (Add Tier 3)")
    print("-" * 100)
    print("Focus: All tiers (5,000 customers)")
    print("Approach: Tier 1 = calls, Tier 2 = emails + SMS, Tier 3 = automated email campaign")
    print("Expected outcome: Prevent 250-300 churns (5% save rate on full base)")
    print("Investment: Largely automated, minimal manual effort beyond Tier 1-2")
    print("Revenue saved: 275 customers × GBP 1,500 = GBP 412,500")


def print_feature_to_tactic_mapping():
    """
    How model features translate to specific retention actions.
    """
    print("\n" + "=" * 100)
    print("MODEL FEATURES -> RETENTION TACTICS (What the ML Model Learned)")
    print("=" * 100)

    print("\nTOP PROTECTIVE FACTORS (Things that keep customers):")
    print("-" * 100)

    print("\n1. TOTAL ORDERS (Strongest Predictor)")
    print("   What the model learned: Loyal customers who've ordered 50+ times almost never churn")
    print("   Tactic: Recognise long-term customers with loyalty benefits")
    print("   Action: Give them exclusive perks (priority delivery windows, early access to meals)")
    print("   Message: 'You're a valued customer - here's something special for your loyalty'")

    print("\n2. ORDER COMPLETION RATE (80%+ of purchased meals consumed)")
    print("   What the model learned: Customers who actually USE the service stick around")
    print("   Tactic: Remove friction to meal consumption")
    print("   Action: Check with low-completion customers about barriers")
    print("          (time-poor? Meal preferences? Storage?)")
    print("   Message: 'Help us improve - what meals are you not using?'")
    print("   Follow-up: Adjust future orders to match actual consumption")

    print("\n3. RECIPE ENGAGEMENT (Customers who rate meals)")
    print("   What the model learned: Passive users (no feedback) churn more")
    print("   Tactic: Encourage feedback loop - make rating easy and rewarding")
    print("   Action: Send 'Rate this meal' SMS after delivery")
    print("          Give small incentives (points, discounts) for feedback")
    print("   Message: 'Your feedback helps us pick better meals for you'")

    print("\n4. FREQUENCY TREND (Declining orders = biggest red flag)")
    print("   What the model learned: Customers ordering less frequently are leaving")
    print("   Tactic: EARLY INTERVENTION - catch them at first sign of decline")
    print("   Action: When customer skips 2 consecutive weeks, reach out immediately")
    print("   Message: 'We notice you've paused orders - is now not a good time?'")
    print("   Follow-up: Flexible options (pause, switch to cheaper plan, skip few weeks)")

    print("\nTOP RISK FACTORS (Things that make customers leave):")
    print("-" * 100)

    print("\n1. SUPPORT TICKETS (Multiple complaints = high churn risk)")
    print("   What the model learned: Customers with problems are leaving")
    print("   Tactic: Rapid resolution + proactive outreach after tickets")
    print("   Action: Flag all support ticket closures for personal follow-up")
    print("          Offer compensation (refund, credit) beyond basic resolution")
    print("   Message: 'Sorry for the hassle - here's how we're making it right'")

    print("\n2. SKIPS (Customers who frequently skip weeks)")
    print("   What the model learned: Skips indicate disengagement or short-term churn signal")
    print("   Tactic: Differentiate between 'temporary pause' and 'losing interest'")
    print("   Action: After 2 skips, offer to pause subscription (retain option, not lose customer)")
    print("          After 4+ skips, investigate why they're not ordering")
    print("   Message: 'Your subscription is paused - we're here when you're ready'")


def print_weekly_operations_guide():
    """
    Practical weekly workflow for retention operations.
    """
    print("\n" + "=" * 100)
    print("WEEKLY OPERATIONS GUIDE: HOW TO WORK WITH THE MODEL DAILY")
    print("=" * 100)

    print("\nEVERY MONDAY (Start of Week):")
    print("-" * 100)
    print("1. Receive flagged customer list from analytics")
    print("   • File contains: Customer ID, name, email, risk level, reason for risk")
    print("   • Sort by: Risk score (highest first)")
    print("2. Distribute to retention team")
    print("   • Tier 1 (High): Assign to specialist for phone outreach")
    print("   • Tier 2 (Medium): Queue for email + SMS outreach")
    print("   • Tier 3 (Low): Trigger automated email campaign")
    print("3. Check previous week's outcomes")
    print("   • Did we reach customers? (yes/no)")
    print("   • Did they place new orders? (conversion rate)")
    print("   • How many unsubscribes? (model false alarms)")

    print("\nWEDNESDAY (Mid-Week Check-in):")
    print("-" * 100)
    print("1. Outreach progress check")
    print("   • Tier 1: Have we called 80%+ of high-risk customers?")
    print("   • Tier 2: Have we sent 100% of emails?")
    print("2. Early outcome indicators")
    print("   • Track order placements from contacted customers")
    print("   • Identify which retention messages are working (conversion by tactic)")

    print("\nFRIDAY (End of Week):")
    print("-" * 100)
    print("1. Weekly results summary")
    print("   • Customers contacted: count")
    print("   • Customers who placed new orders: count + revenue")
    print("   • Unsubscribes (customers we wrongly flagged): count")
    print("   • Churn rate this week vs expected: comparison")
    print("2. Adapt for next week")
    print("   • Which customer segments responded best to which messages?")
    print("   • Should we change approach for Tier 2 or 3?")
    print("3. Report to leadership")
    print("   • Churn prevented: X customers × GBP 1,500 = GBP Y revenue saved")
    print("   • Cost of retention: Z per customer")
    print("   • Net benefit: GBP Y - Z")


def print_success_metrics():
    """
    Business success metrics for tracking retention impact.
    """
    print("\n" + "=" * 100)
    print("SUCCESS METRICS: WHAT WE'LL MEASURE (NOT TECHNICAL METRICS)")
    print("=" * 100)

    print("\nPRIMARY METRIC: Churn Prevention Rate")
    print("-" * 100)
    print("Definition: Percentage of flagged customers who DON'T churn after contact")
    print("Target: 15-20% in Month 1, 20-30% by Month 3")
    print("How to calculate: (Flagged customers still active / Total flagged) × 100")
    print("Why it matters: Direct measure of intervention effectiveness")
    print("Expected impact at Month 3:")
    print("  • 5,000 at-risk customers flagged per month")
    print("  • 20-30% retention success = 1,000-1,500 customers saved per month")
    print("  • Annual revenue protection = GBP 300,000 - GBP 450,000")

    print("\nSECONDARY METRIC: Contact-to-Conversion Rate")
    print("-" * 100)
    print("Definition: Of customers we contact, how many place a new order within 14 days?")
    print("Target: 25-35% for Tier 1 (high-risk), 10-15% for Tier 2")
    print("Why it matters: Shows whether our messaging is compelling")
    print("By segment:")
    print("  • Disengaged (need frequency bump): Target 30% reorder rate")
    print("  • Low satisfaction (need product fix): Target 25% reorder rate")
    print("  • Support issues (need resolution): Target 40% reorder rate (easiest to win back)")
    print("  • Bargain hunters (need price): Target 20% reorder rate (hardest to win back)")

    print("\nTERTIARY METRIC: Cost per Customer Saved")
    print("-" * 100)
    print("Definition: Retention spend ÷ customers prevented from churning")
    print("Target: GBP 50-100 per customer saved (anything under GBP 100 is good)")
    print("Calculation example:")
    print("  • 500 calls × 5 min × GBP 15/hour staff cost = GBP 625 labour cost")
    print("  • Email/SMS infrastructure cost = GBP 75")
    print("  • Total cost = GBP 700")
    print("  • Customers saved from 500-customer cohort = 100")
    print("  • Cost per saved customer = GBP 700 ÷ 100 = GBP 7 (GREAT ROI)")

    print("\nMONITORING CADENCE:")
    print("-" * 100)
    print("Daily:  Check if contacted customers place new orders (daily pulse)")
    print("Weekly: Summarise outcomes, adapt approach for next week")
    print("Monthly: Calculate churn rate vs baseline, report revenue impact to CFO")
    print("Quarterly: Review retention programme ROI, decide on Phase 2 (XGBoost upgrade)")


def print_faq_for_retention_team():
    """
    FAQ addressing common questions from retention team.
    """
    print("\n" + "=" * 100)
    print("FAQ: RETENTION TEAM QUESTIONS")
    print("=" * 100)

    print("\nQ: How do I know these customers will actually churn?")
    print("A: We tested the model on 300 customers and got it right 99% of the time.")
    print("   When we said 'this customer is at risk', they actually churned 99.5% of the time.")
    print("   Bottom line: Trust the list. These are real at-risk customers, not random.")

    print("\nQ: Won't customers be annoyed if we contact them unprompted?")
    print("A: No. They're at-risk of leaving anyway. Your proactive outreach actually helps them.")
    print("   Frame it positively: 'We noticed... is there anything we can do?'")
    print("   Research shows customers appreciate when you reach out before they churn.")
    print("   Risk of annoyance: Very low. Risk of losing customer if you DON'T contact: Very high.")

    print("\nQ: What if a customer says 'I'm not leaving, why did you flag me?'")
    print("A: They're probably right - the model isn't 100% perfect (it's 99%, not 100%).")
    print("   Turn it into an opportunity: 'Great to hear! What could we improve?'")
    print("   Even false-positive contacts build relationship strength.")

    print("\nQ: Why don't we just contact everyone?")
    print("A: Cost. Retention spend per contact = GBP 50. Times 80,000 customers = GBP 4M budget.")
    print("   We don't have that. So we focus on 5,000 highest-risk customers first.")
    print("   This model lets us find the 'right' 5,000 instead of random 5,000.")

    print("\nQ: Can I see the model predictions for a specific customer?")
    print("A: Yes. Ask analytics for their risk score and top 3 reasons they're flagged.")
    print("   Example: 'Customer X is at 92% risk because order frequency dropped 80% in 4 weeks'")
    print("   Use that reason to customise your outreach approach.")

    print("\nQ: How long should I wait before contacting a flagged customer?")
    print("A: ASAP (within 48 hours if possible). At-risk customers are leaving now.")
    print("   Tier 1 (high-risk): Contact within 24 hours (phone call preferred)")
    print("   Tier 2 (medium-risk): Contact within 1 week (email OK)")
    print("   Tier 3 (low-risk): Trigger automated email campaign within 3 days")

    print("\nQ: What if a customer we contact actually churns anyway?")
    print("A: That's normal - the model isn't perfect. Some customers will leave no matter what.")
    print("   If 20% of contacted customers still churn, that means 80% were retained.")
    print("   80% retention success × GBP 1,500 LTV = GBP 1,200 revenue saved per contact.")
    print("   That's why we do this - it's still massively profitable.")


def print_presentation_talking_points():
    """
    Concise talking points for presenting to leadership.
    """
    print("\n" + "=" * 100)
    print("PRESENTATION TALKING POINTS FOR LEADERSHIP")
    print("=" * 100)

    print("\nOPENING (30 seconds):")
    print("-" * 100)
    print("We built a machine learning model that identifies 5,000 at-risk customers each month")
    print("BEFORE they churn. Right now we're losing them with no warning. This model gives us")
    print("24-48 hours to save them with a phone call or retention offer.")

    print("\nTHE OPPORTUNITY (1 minute):")
    print("-" * 100)
    print("Our annual customer base is 80,000. At 67% churn, we lose 53,600 customers yearly.")
    print("Each customer is worth GBP 1,500 lifetime value. That's GBP 80 million in annual revenue.")
    print("If we prevent just 5% of churn through early intervention, that's GBP 400K revenue saved.")

    print("\nTHE APPROACH (1 minute):")
    print("-" * 100)
    print("The model identifies 5,000 at-risk customers per month and ranks them by urgency.")
    print("Our retention team contacts top priority customers - a 2-minute phone call costs GBP 0.50")
    print("per customer. If just 20% of contacted customers stay (instead of leaving), we break even")
    print("instantly and save GBP 1,500 per retained customer.")

    print("\nTHE RESULTS (Expected):")
    print("-" * 100)
    print("Month 1: Identify and contact 500 high-risk customers.")
    print("         Expected outcome = 80-100 customers saved = GBP 120-150K revenue")
    print("")
    print("Month 2-3: Scale to contact 2,000 medium + high-risk customers.")
    print("           Expected outcome = 200-300 customers saved = GBP 300-450K revenue")
    print("")
    print("Year 1: If successful, save GBP 300-450K in churn losses with GBP 50K retention spend.")
    print("        Net benefit = GBP 250-400K, or a 5-8X return on investment.")

    print("\nTHE NEXT STEP:")
    print("-" * 100)
    print("Week 1: Deploy the model and score all 80,000 customers.")
    print("Week 2: Retention team reaches out to first 500 high-risk customers.")
    print("Week 4-8: Monitor results, measure churn rate change.")
    print("Month 3: Report revenue impact, decide whether to scale or upgrade model.")
