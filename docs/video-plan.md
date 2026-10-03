# Video plan — prototype demonstration

Target **11 minutes** (the limit is 20; a tight video reads as confident). Record in one take if you can, with the app already running and the model already loaded so there is no waiting.

---

## Before you press record

- [ ] `streamlit run app/app.py` already running, one photograph already put through it, so the model is loaded and the next one is instant.
- [ ] `models/test_metrics.json` in place, so the sidebar shows the **test** numbers.
- [ ] Browser zoom at 125–150 %. Anything at normal zoom is unreadable once the video is compressed.
- [ ] Do Not Disturb on. No notification banners mid-demo.
- [ ] Close unrelated tabs; the report PDF and the figures open in another window, ready to switch to.
- [ ] A glass of water and one practice run. The first take is always 3 minutes too long.

Record with Shift-Cmd-5 → *Record Selected Portion* (not the whole screen — a cropped 16:9 area keeps the text large), microphone on.

---

## Running order

**1. What this is — 0:00–0:45**
On screen: the app's title page.
Say: what diabetic retinopathy is in two sentences; that the system answers two questions (is there DR, and which stage); that it is an academic prototype, not a medical device, trained on the public APTOS 2019 dataset. Get the disclaimer out early and in your own voice — it sets the tone for everything after.

**2. The dataset and its two traps — 0:45–2:00**
On screen: fig01, then fig12.
Say: 3,662 photographs, half of them healthy, only 5 % severe. Then the trap that shaped the whole project: the image size tells you which camera took the photograph, and one camera's photographs are 93 % healthy while another's are almost all diseased. A model that only knows the image size gets DR yes/no right 89 % of the time. Say that this is why there is a camera-only baseline on every chart. Mention in one line that 173 images were dropped as duplicates, 77 of them because copies of the same photograph carried different stages.

**3. Preprocessing — 2:00–3:15**
On screen: fig13, then fig14.
Say: find the retina, crop to it, resize to 512, then give every photograph the *same* window — full width, top and bottom trimmed flat — because some cameras cut the retina off and that shape alone nearly separates sick from healthy. Then the three variants, and that the choice between them was an experiment, not a preference.

**4. Augmentation and balance — 3:15–4:00**
On screen: fig20, then fig21.
Say: flips and any rotation are realistic because a fundus photograph has no "up" and left and right eyes mirror each other; colour shifts were left out because lesion colour carries meaning. Augmentation runs on the training split only. Then the two balancing options and the fact that neither beat doing nothing by more than the run-to-run noise.

**5. The model — 4:00–5:15**
On screen: fig36, then fig24.
Say: EfficientNetB3 pretrained on ImageNet, a small head, two phases — head first with the backbone frozen, then the top quarter of the layers fine-tuned at a tenth of the learning rate, with BatchNorm kept frozen. Point at the curves: training accuracy pulls away from validation after fine-tuning starts, which is why the kept model is the best validation macro-F1, not the last epoch.

**6. Experiments and results — 5:15–7:00**
On screen: fig22, then the test results table, then fig31.
Say: 15 runs, two seeds each, everything judged on validation only. Enhancement helped a little, balancing didn't clear the noise, fine-tuning helped everywhere, B3 won. Then the test set, used once: accuracy 82 %, macro-F1 0.62, QWK 0.90, and for the yes/no question sensitivity 95 % and specificity 99 % — 12 diseased eyes missed out of 255. Then fig31, and the sentence that matters: on the one camera whose photographs give nothing away, the model gets 82 % while the camera-only model gets 36 %.

**7. The prototype, live — 7:00–9:30**
This is the part that is being marked as a demonstration. Show four photographs, in this order:

| Photograph | Why show it |
|---|---|
| `901a3552fe26` (true 0) | A clean healthy eye: answer No DR, 100 %. Point out the three images — original, cropped to the common window, and the CLAHE version the model actually sees. |
| `b6a0e348a01e` (true 1) | Mild caught correctly at 66 % — show the probability bar chart and say that the model is much less certain here. |
| `70ed3ec68b94` (true 2) | Moderate at 97 %; open the Grad-CAM panel and explain that the heat sits on the retina, and that across the whole test set 96 % of it does. |
| `a182b5b191de` (true 4) | **Show the failure.** The model is certain DR is present but splits the stage 0.40 / 0.31 / 0.29 and answers Moderate for a proliferative eye. Say exactly that: detection is strong, grading is not. |

If you have time, add `6f4719c6bb4b` to show the quality warning firing on a dark photograph. Say what the warnings are built from — the brightness, contrast and sharpness range of the training photographs.

Being the one to point out your own failure case is worth more than a clean demo.

**8. Limitations and ethics — 9:30–10:30**
Say, without hedging: one dataset, one region, a few cameras; labels from a single grader; no patient IDs so we cannot rule out the same eye appearing twice; Severe has 26 test images and its F1 could be anywhere between 0.11 and 0.47; the model under-grades severe disease, which is the direction that would matter clinically. Then what would be needed before anything like this could be used: prospective validation, more than one population, regulation, and a defined role for the clinician.

**9. Close — 10:30–11:00**
On screen: the repository.
Say: four notebooks, three shared modules, the split saved once and used by all of them, the test set touched once. What you would do next: class weights with B3, a second seed, and an "image not gradable" output.

---

## Hosting

1. Export the recording (Shift-Cmd-5 saves to the Desktop by default).
2. Upload to YouTube as **Unlisted** — not Private, which nobody else can open.
3. Open the link in a private browser window to confirm it plays without signing in.
4. Put the URL in the report: on the title page and in section 8. Check it is clickable in the exported PDF.
5. Keep the video file until after the marks are released.

---

## If you record in pieces

Record sections 1–6 in one take and 7–9 in another, then join them. Don't try to edit within a take; re-record it. Keep every take, and name them `take1_intro.mov` so you don't lose the good one.
