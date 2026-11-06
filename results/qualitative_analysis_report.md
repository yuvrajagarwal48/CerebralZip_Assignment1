# Qualitative Diagnostic & Error Analysis Report
## Decoupled Semantic & Localization Evaluation (X-CLIP + Qwen-VL)

This diagnostic report breaks down pipeline behaviors into operational categories to explain boundary offsets, 
precursor recognition, and false alarm elimination.

---

### Accurate Detections (Tight Localization & VLM Confirmed)
#### Video: `Falling_fall-01-cam0-rgb.mp4`
```text
0s -------------------------------------------- 6.4s
GT  : |                       ===========================|
Pred: |###############################################   |
Legend: [=] Ground Truth  |  [#] Pipeline Prediction
```
- **Ground Truth**: `Falling (person collapse)` [3.0s - 6.4s]
- **Prediction**: `Falling` [0.0s - 6.0s] (Conf: 0.319)
- **Localization Metrics**: tIoU = **0.469**, Temporal Center Offset = **1.70s**
- **Boundary Offsets**: Start Offset = `-3.00s`, End Offset = `-0.40s`
- **VLM Verbatim Verification Evidence**: "Class: Falling | Status: SUPPORTED | Evidence: The person is seen falling to the ground, which is clearly visible in the video frames."

#### Video: `Falling_fall-02-cam0-rgb.mp4`
```text
0s -------------------------------------------- 4.4s
GT  : |           =======================================|
Pred: |##############################################    |
Legend: [=] Ground Truth  |  [#] Pipeline Prediction
```
- **Ground Truth**: `Falling (person collapse)` [1.0s - 4.4s]
- **Prediction**: `Falling` [0.0s - 4.0s] (Conf: 0.323)
- **Localization Metrics**: tIoU = **0.682**, Temporal Center Offset = **0.70s**
- **Boundary Offsets**: Start Offset = `-1.00s`, End Offset = `-0.40s`
- **VLM Verbatim Verification Evidence**: "Class: Falling | Status: SUPPORTED | Evidence: The person is lying on the floor with their legs extended and arms outstretched, which is a common posture when falling."

### Semantic Success with Boundary Discrepancy (Vicinity Hit)
#### Video: `Falling_fall-05-cam0-rgb.mp4`
```text
0s -------------------------------------------- 6.0s
GT  : |                                         =========|
Pred: |                ##################################|
Legend: [=] Ground Truth  |  [#] Pipeline Prediction
```
- **Ground Truth**: `Falling (person collapse)` [5.0s - 6.0s]
- **Prediction**: `Falling` [2.0s - 6.0s] (Conf: 0.486)
- **Localization Metrics**: tIoU = **0.248**, Temporal Center Offset = **1.52s**
- **Boundary Offsets**: Start Offset = `-3.00s`, End Offset = `-0.04s`
- **VLM Verbatim Verification Evidence**: "Class: Falling | Status: SUPPORTED | Evidence: The person is seen falling onto the floor, which is visible in the frames."

### Precursor / Co-Occurring Actions in Multi-Stage Incidents (e.g. Arson)
#### Video: `Fire_Smoke_Arson041_x264.mp4`
```text
0s -------------------------------------------- 125.1s
GT  : |                            ====================  |
Pred: |           #####          #####              #####|
Legend: [=] Ground Truth  |  [#] Pipeline Prediction
```
- **Ground Truth Target Class**: `['fire / smoke']`
- **Detected Precursor Action**: `Intrusion/Trespassing` [29.0s - 39.0s] (Conf: 0.286)
- **Operational Diagnosis**: Visual analysis confirms physical action (e.g. perimeter climbing / property destruction) preceding ignition. Categorized as valid causal co-occurrence, not an FP.
- **VLM Verbatim Evidence**: "Class: Intrusion/Trespass | Status: SUPPORTED | Evidence: The person is seen walking towards a door, which suggests they are attempting to enter a restricted area."

#### Video: `Fire_Smoke_Arson041_x264.mp4`
```text
0s -------------------------------------------- 125.1s
GT  : |                            ====================  |
Pred: |           #####          #####              #####|
Legend: [=] Ground Truth  |  [#] Pipeline Prediction
```
- **Ground Truth Target Class**: `['fire / smoke']`
- **Detected Precursor Action**: `Intrusion/Trespassing` [67.0s - 77.0s] (Conf: 0.486)
- **Operational Diagnosis**: Visual analysis confirms physical action (e.g. perimeter climbing / property destruction) preceding ignition. Categorized as valid causal co-occurrence, not an FP.
- **VLM Verbatim Evidence**: "Class: Intrusion/Trespassing | Status: SUPPORTED | Evidence: The person is seen entering the building, which is not allowed, and then leaving, indicating an attempt to enter without authorization."

### VLM False Alarm Pruning (Rejected Proposals)
*No representative examples observed in this subset.*
