BANAO TECHNOLOGIES — TASK 2 SUBMISSION FORM

1. WHAT DID YOU TRY?

I built a complete service-request routing system for Kestrel Home Appliances.

I joined the training requests with the resolution log using request_id, using final_team — the team that actually resolved each ticket — as the training target. I normalized the legacy team names according to the operations policy: “Installations” became “Installs & Demo” and “Consumables” became “Filters & Consumables.”

I also audited the legacy routing bot by comparing its normalized team_label against the actual final_team. The audit found 2,471 misroutes out of 10,822 comparable tickets, an error rate of 22.83%. At Rs 565 per misroute, consisting of Rs 305 in transfer cost and Rs 260 in additional customer-contact cost, this represents Rs 13,96,115 in historical misrouting cost.

For the replacement system, I evaluated a local TF-IDF + LinearSVC approach and optimized the text representation. The final verified model uses character-level TF-IDF with char_wb n-grams (3,6), sublinear_tf=True, and LinearSVC.

Using a fixed stratified 80/20 holdout, the final model achieved 84.71% accuracy, 0.8557 macro F1, and 0.8492 weighted F1.

The model was then retrained on all 10,822 valid labeled records and saved as routing_model.pkl.

I also built a FastAPI service with /health and /predict, a frontend for testing individual requests, and generated predictions for all 2,178 test requests.


2. WHAT DID YOU CHANGE?

The most important change was the training target.

The task described a goal of approximately 90% agreement with the legacy routing bot. After comparing the bot's labels with actual resolution outcomes, I found that the legacy bot was wrong on 22.83% of tickets.

Rather than train a new system to reproduce those errors, I changed the target from team_label to final_team, so the model learns from actual resolution outcomes.

I also tested alternative text representations and selected the best verified configuration while keeping the evaluation set untouched.


3. WHAT DID YOU DISCARD?

I discarded the idea of optimizing primarily for agreement with the legacy bot after the evidence showed that the legacy bot was unreliable against actual resolution outcomes.

I also evaluated additional request-time structured fields such as product, channel, and warranty information, but the text-only approach provided the strongest verified result under the evaluated setup.

I did not add an LLM or external inference API because the problem can be solved locally with a lightweight supervised model, avoiding external API cost and dependency.


4. WHAT DID YOU DELIBERATELY NOT BUILD?

I deliberately did not build an LLM-based classifier because it would add API cost, latency, and external dependency.

I did not build an automated retraining pipeline because this is an initial deployment and there is not yet enough production history to justify automated retraining and drift handling.

I also did not build a live operational analytics/metrics dashboard. I did build a lightweight triage UI (index.html) for support agents to test individual requests, while aggregate monitoring and drift analytics can be added after production deployment.


5. WHAT DID YOU BUILD THAT WAS NOT EXPLICITLY REQUESTED?

I added:

- Detailed validation and error analysis to understand routing mistakes.
- A human-readable reason field in the /predict API response.
- Legacy-bot misrouting analysis and financial impact calculation.
- A local support-agent triage interface.
- A clean README with reproducibility and handover instructions.


6. WHAT ARE THE THREE MOST IMPORTANT THINGS ANOTHER ENGINEER NEEDS TO KNOW IF THEY TAKE OVER ON MONDAY?

1. The model predicts final_team, not team_label. The value of this system depends on learning from actual resolution outcomes rather than reproducing the legacy bot's assignments.

2. Team-name normalization is required. The operations policy renamed “Installations” to “Installs & Demo” and “Consumables” to “Filters & Consumables.” These names must be normalized consistently before training and comparison.

3. The test set was not used for training or validation. The reported 84.71% accuracy comes from a held-out 20% stratified validation split. The 2,178 test requests were kept separate for final prediction generation.


7. HOW MANY HOURS DID THE TASK ACTUALLY TAKE?

Approximately  5 hurs, including exploratory data analysis, model development, optimization, API development, testing, GitHub preparation, and documentation.

8. WHAT SCORE DO YOU EXPECT THE HIDDEN predictions.csv TO ACHIEVE, ON WHAT METRIC, AND WHY?

The final model achieved 84.71% accuracy on the untouched validation holdout against actual final_team outcomes, with a macro F1 of 0.8557.

For the hidden evaluation, accuracy is the primary metric because this is a multi-class team-routing problem where each request receives one final team label.

The hidden-test result is not known until Banao evaluates predictions.csv, so I will not claim a hidden-test score in advance.


9. HOW DO YOU KNOW IT WORKS?

The model was evaluated using a reproducible stratified 80/20 split, with the 20% holdout kept separate from model selection and tuning.

The final verified holdout results were:

Accuracy: 84.71%
Macro F1: 0.8557
Weighted F1: 0.8492
Holdout errors: 331 of 2,165

The API was tested directly with multiple customer requests, the frontend was verified through the running service, and predictions.csv was checked for schema, row count, missing values, and valid team names.


10. DID YOU CHANGE, NARROW, OR PUSH BACK ON THE CLIENT'S ASK? WHAT, WHEN, AND WHY?

Yes.

After joining the intake data with actual resolution outcomes, I found that the legacy routing bot had an accuracy of only 77.17%, with 2,471 misroutes and an estimated historical cost of Rs 13,96,115 at Rs 565 per misroute.

I therefore reassessed the requested 90% agreement target rather than treating agreement with the legacy bot as the primary definition of success.

The replacement model was trained against final_team, the actual resolution outcome, rather than team_label, the legacy bot assignment.

The objective was to improve routing quality against the real operational outcome rather than reproduce an existing system's known errors.


11. WHAT IS WRONG WITH WHAT YOU ARE HANDING OVER?

The final model achieves 84.71% validation accuracy, so it still makes some routing errors.

The remaining errors include overlap between service categories where customer language can be ambiguous.

The current implementation is also a first-deployment system and does not yet include automated production drift monitoring or an automated retraining workflow.

The model runs locally with zero paid inference API cost, but future hardware and infrastructure costs are outside this calculation.


12. WHAT DID YOU DELIBERATELY LEAVE OUT, AND WHY?

I left out:

- External LLM inference
- Automated model retraining
- Full operational analytics and drift monitoring

These were intentionally excluded to keep the first deployment small, reliable, offline-capable, and focused on the core routing requirement.


13. WHAT AI TOOLS DID YOU USE?

I used ChatGPT and Antigravity IDE with Gemini/Claude-assisted coding and reasoning for implementation support, debugging, analysis, documentation, and review.

The final code, metrics, predictions, and API behavior were executed and verified locally rather than being accepted solely from AI-generated output.


14. WHAT DID YOU TRY, WHAT DID YOU CHANGE, AND WHAT DID YOU DISCARD?

I started with a word-level TF-IDF + LinearSVC baseline and obtained 83.23% validation accuracy.

I then tested legitimate hyperparameter changes and alternative text representations. Character-level TF-IDF improved the verified holdout result to 84.71%, which became the final configuration.

I also evaluated additional structured request-time fields, but they did not provide sufficient improvement to justify replacing the simpler text-only production approach.

I discarded the idea of training against team_label because the legacy bot was demonstrably unreliable against actual resolution outcomes.


15. WHAT DID YOU BUILD THAT NOBODY EXPLICITLY ASKED FOR?

I added:

- Detailed legacy-bot misrouting analysis
- Error-pattern analysis
- A human-readable routing reason in the API
- A local support-agent triage interface
- A reproducible README and Monday handover documentation

These additions were intended to make the system easier to evaluate, operate, and hand over.


16. WHAT IS THE PREDICTION COST AND WHAT WOULD IT COST KESTREL'S VOLUME?

The current model runs locally without a paid inference API.

Prediction cost: Rs 0.00 per prediction.

For the 2,178 test requests generated for this submission, the direct model inference cost is therefore Rs 0.00.

At production volumes, the direct software inference cost remains Rs 0.00 per prediction. Hardware and general infrastructure costs are not included in this calculation.


17. REPOSITORY / SUBMISSION LINKS

GitHub repository:
https://github.com/AakashDubba/kestrel-service-routing




18. KEY VERIFIED METRICS

Training records: 10,822
Validation records: 2,165
Final classes: 7
Old bot accuracy: 77.17%
Old bot error rate: 22.83%
Old bot misroutes: 2,471
Cost per misroute: Rs 565
Historical misrouting cost: Rs 13,96,115
Final model accuracy: 84.71%
Macro F1: 0.8557
Weighted F1: 0.8492
Validation errors: 331
Test predictions: 2,178
Prediction cost: Rs 0.00
Monthly model/software cost: Rs 0.00
Annual legacy bot licence: Rs 3,20,000


19. HOW TO RUN

pip install -r requirements.txt
python train_and_predict.py
uvicorn app:app --host 0.0.0.0 --port 8000

Open:
http://localhost:8000