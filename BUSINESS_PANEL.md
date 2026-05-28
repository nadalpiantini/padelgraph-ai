# Business Panel Critique — padelgraph-ai (2026-05-28)

> Generated via Sephirot `business-panel-experts` agent in **debate mode**.
> Captured verbatim. Cross-referenced in `.decision-log.md` entry D12.

DEBATE MODE. 5 voices. No sycophancy. Steel-manned where the case deserves it, dismantled where it doesn't.

---

## Round 1 — Opening positions

**Drucker.** Let us start where every founder forgets to start: *what business are you in?* You have shipped a Python computer vision library. That is a tool, not a business. The repo is Apache 2.0, the YOLOX loader is still a stub, the CLI takes two videos plus a calibration JSON. Who is the customer for this artifact today? A padel club operator cannot install Kornia. A coach will not write homography matrices. A developer at PlaySight might fork it — and owe you nothing. Until you can name the customer in one sentence and the result they value in another, this is a hobby with good engineering. The fact that the business model is listed as five options (a) through (e) is itself the finding — it means there is no business model, only optionality fantasy. Pick one before Epic 2.

**Christensen.** I want to defend the founder slightly, then sharpen the knife. The job-to-be-done in padel video is real and undershot: amateur players want to *see themselves play and improve*, clubs want to *justify court premium with content*, and federations want *ranking-grade data without Hawkeye budgets*. Veo's absence from padel is genuine non-consumption — that part of the thesis holds. But here is the disruption question the founder is dodging: disruption requires a *value network* where the new entrant's "worse on traditional metrics, better on new metrics" trade is acceptable. Padel clubs today buy nothing or buy a $40 phone mount. The new metric padelgraph-ai must win on is *cost-per-match-analyzed approaching zero through OSS*. If it doesn't, it's just a worse PlaySight.

**Kim/Mauborgne.** The brief invited skepticism about blue ocean and I'll honor it: **this is not blue ocean.** Padel CV is a contested red ocean with Pixellot, PlaySight, Mojjo (yes, dead — but their corpse is still landing rights), Wingfield, and a long tail of phone-mount apps. The fact that no Apache/MIT padel repo has >300★ is not evidence of blue ocean; it is evidence that *the OSS strategy specifically has been tried and abandoned*. Ask why. Likely answer: padel CV requires multi-camera calibration that is impossible to crowdsource — every court is different, every install is bespoke, every dataset is proprietary. The eliminate-reduce-raise-create exercise here should *eliminate* the assumption that "OSS reference implementation" is a defensible position. It is a gift to whoever monetizes downstream — and that will not be Alan.

**Taleb.** Apache 2.0 on a single-founder LATAM project with no moat and no revenue is the textbook *negative-optionality* trade. You absorb all the downside (12 months of solo work, opportunity cost vs Empleaido which is closer to revenue, reputation risk if the stub loaders never get filled), and you cap your upside at "acquihire" or "consulting on top of OSS" — both of which require a *buyer who already knows you exist*. Where is your asymmetry? I see fragility: one founder, GPU-less mesh, cloud LLM dependency, a market dominated by hardware-bundle incumbents with capex you cannot match. The antifragile move in this space is *not the pipeline*. It is the dataset. Annotated padel footage, court calibration profiles, player trajectory ground truth — those compound. Code depreciates; data appreciates. The founder built the depreciating asset first.

**Porter.** Let me name the five forces because nobody has: **(1) Rivalry** — moderate-to-high, hardware incumbents with distribution. **(2) Substitutes** — extreme: phone + free app + human coach beats any analytics product for 95% of amateurs. **(3) Buyer power** — high: padel clubs are price-takers, federations are bureaucratic and slow. **(4) Supplier power** — low (good — open models, cheap compute). **(5) Barriers to entry** — *the founder is lowering them by going Apache 2.0*. This is strategically incoherent. You cannot simultaneously be the OSS commons *and* extract rent. Pick: differentiation (vertical SaaS on top, padel-specific dataset moat) or cost leadership (cheapest cloud per-match credit, which requires scale you don't have). Right now you've picked neither and called it "TBD."

---

## Round 2 — Disagreements

**Taleb → Christensen.** Clayton, your disruption frame assumes the incumbent is asleep. Pixellot has 38,000 systems deployed and a sales team. The moment padelgraph-ai shows traction, Pixellot ships a padel SKU in 90 days. The Apache license means they can fork it. Disruption theory needs a *protected runway* — what is yours?

**Christensen → Taleb.** Fair, but you're modeling Pixellot as rational. They're not — they're an installed-base business optimizing for tennis/soccer/basketball where the unit economics work. Padel is a rounding error to them until it isn't. The runway is 18-24 months of incumbent inattention. The real question is whether Alan can convert that runway into *something other than code* — and on that, I agree with you and Porter both.

**Porter → Kim/Mauborgne.** You called this red ocean and I want to push back partially. The *padel analytics* ocean is red. The *padel-specific computer vision toolkit that lets others build vertical products* sub-segment is genuinely empty. The question isn't whether the ocean is blue — it's whether that sub-segment has a buyer with budget. I don't think it does. Toolkit-layer OSS only monetizes via (a) hosted version, (b) enterprise support, (c) dual license. Apache 2.0 forecloses (c).

**Kim/Mauborgne → Porter.** Agreed. And Apache 2.0 specifically is the founder's most expensive *unforced* decision. If you must OSS, AGPL with commercial dual-license preserves option C. Apache gives the moat away on day one — to anyone, forever. That isn't strategy. That is a community-signaling gesture priced at the cost of the entire downstream market.

**Drucker → all.** You are all arguing tactics. None of you has asked the question I asked: *what business?* If the answer is "OSS reference implementation that builds Alan's reputation as the LATAM AI cinema/sports infrastructure builder," then Apache 2.0 is correct, the lack of revenue is correct, and the success metric is *inbound from Filmatron-adjacent or sports-fed contracts*, not GitHub stars. But nobody — including the founder — has stated this. So we are all critiquing a strategy that doesn't exist.

**Taleb → Drucker.** This is the most useful thing said so far. The optionality only becomes positive if padelgraph-ai is a *reputation asset* feeding higher-margin work (consulting, Filmatron credibility, LATAM sports federation contracts). As a standalone product, it is dead on arrival. As a calling card, it might be brilliant — but only if Alan stops treating it as a product.

---

## Round 3 — Steel-manning the founder's case

**Christensen.** Steel-manned: Alan is the *only* person globally who has shipped a working multi-camera padel CV pipeline as OSS in a market with 25M players and zero LATAM incumbent. If even 1% of LATAM's 84K projected courts (2026) want any form of analytics, and Alan is the default "padel CV guy" on GitHub, inbound consulting alone justifies the 12 months. This is real.

**Porter.** Steel-manned: by going Apache 2.0 first, Alan trades extraction for *category definition*. If padelgraph-ai becomes the de facto schema for padel match data (MatchAnalysis JSON), every downstream product — clubs, federations, hardware bundlers — interoperates through his format. That is a *standards-setting* play, not a product play. Worth more than any SaaS he could build solo.

**Kim/Mauborgne.** Steel-manned: the blue ocean is not the pipeline. It is *LATAM-Spanish-language padel data infrastructure* with first-mover credibility. Pixellot won't translate documentation. PlaySight won't court Mexican federations. There's a regional moat in language, relationships, and time-zone that no incumbent will replicate cheaply.

**Taleb.** Steel-manned: Apache 2.0 + public repo is *cheap optionality if the founder pivots to dataset/services*. The code is bait. If 50 clubs install it and contribute calibration profiles, Alan owns the only padel calibration dataset on Earth. That is the antifragile asset. Possible but requires deliberate community-building Alan has not shown evidence of doing.

**Drucker.** Steel-manned: the *business* might be Alan Nadal Piantini as a brand — Sephirot + Filmatron + Empleaido + padelgraph-ai as a portfolio that says "this person ships AI infrastructure in domains nobody else covers from LATAM." padelgraph-ai is then a *credentialing artifact*. Coherent. But say so out loud, and stop pretending it's a product.

---

## Synthesis

### 3 weakest assumptions
1. **"OSS reference implementation is a defensible position."** It is not. Apache 2.0 with no hosted/enterprise/dataset layer gives the entire downstream market to whoever monetizes first — almost certainly not the founder.
2. **"Veo's absence = market opening for me."** Veo's absence reflects padel's installed base economics, not opportunity. The actual incumbents are Pixellot, PlaySight, and the phone+mount stack — and the buyer (clubs) has near-zero analytics budget.
3. **"Business model TBD is fine for now."** It is not. Apache 2.0 was chosen *before* the model. That ordering forecloses options (dual-license, proprietary dataset, hosted enterprise) that cannot be reopened.

### 3 most likely failure modes
1. **Zombie repo.** 12 months in, stars plateau under 200, no contributors, Epic 2 (real YOLOX, BlazePose) never fully ships because solo capacity goes to Empleaido. D24 retrospect fails.
2. **Successful but extracted by others.** A hardware bundler (BLAUBECK, mount maker, regional Pixellot reseller) wraps padelgraph-ai, sells the bundle, captures the margin. Alan gets a GitHub thank-you and zero revenue.
3. **Strategic drift.** padelgraph-ai consumes the one-project-at-a-time slot (D2) that Empleaido — the project actually approaching revenue — needs for push-mode. Opportunity cost compounds.

### 1 missing path forward
**Stop calling it a product. Reframe as a 12-month standards + reputation play, and build the dataset moat the code itself cannot provide.** Specifically: ship the pipeline as the bait (keep Apache 2.0 — it's the right license *for this purpose*), but spend the real effort on (a) defining MatchAnalysis JSON as the open schema for padel data, (b) signing 3-5 LATAM clubs/federations as design partners who contribute calibration profiles and annotated footage to a *separately licensed* dataset (CC-BY-NC or proprietary), and (c) explicitly positioning Alan as the LATAM padel data infrastructure person — which feeds Filmatron's sports-cinema angle and creates inbound consulting. The code is depreciating. The schema, the dataset, and the relationships are appreciating. The founder built the wrong asset first; reverse the priority while the code investment is still fresh.

### Verdict
**Needs strategic reframe within 60 days — keep shipping code, but treat it as bait for the dataset+schema+reputation play, not as a product. Otherwise: pivot or retire by month 6, not month 12.**
