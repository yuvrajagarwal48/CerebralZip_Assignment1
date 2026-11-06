# Zero-Shot Activity Detection and Temporal Localization in Long Videos

## 1. Introduction

The objective of this project is to detect unusual activities in long, untrimmed videos and identify when each activity occurs. The system is expected to recognize activities from a predefined set of classes and produce their start time, end time, class label and confidence score.

The target activities include Fighting/Physical Assault, Falling, Fire/Smoke, Intrusion/Trespassing, Loitering, Vandalism, Theft/Chain Snatching, Crowd Formation, Running in a Restricted Zone and Abandoned Object.

The main challenge is that these activities can occur briefly within long videos. A system must therefore recognize the activity as well as identify its approximate temporal boundaries.

## 2. Approach

### Initial approach

The initial implementation was based on a video-language model (VLM) that analysed videos in overlapping temporal windows and generated descriptions of the activities it observed. These descriptions were then passed to a language model to combine observations, remove duplicates and assign activity labels. A final VLM verification stage was intended to check the predicted activities against the original video.

Although this approach could describe general human movement, it struggled to consistently identify specific actions. For example, activities such as kicking or other short physical interactions were sometimes missed or described only as ordinary human movement. The generated observations also contained considerable background activity, which made classification and temporal localization more difficult.

### Revised approach: X-CLIP with VLM verification

To address these limitations, the pipeline was redesigned around zero-shot video-text similarity using X-CLIP.

Instead of first generating descriptions of everything happening in a video, X-CLIP directly compares video content with natural-language descriptions of the target activities.

The revised pipeline consists of the following stages:

**Video processing and temporal segmentation**

Long videos are divided into overlapping temporal windows. Frames are extracted from each window and passed to X-CLIP. Overlapping windows help reduce the possibility of missing activities occurring near window boundaries.

**Activity prompt matching**

Each target class is represented using multiple textual descriptions. For example, Fighting/Physical Assault includes descriptions of punching, kicking, pushing and wrestling.

A shared background prompt list is also used to represent ordinary activities such as walking, waiting, talking and normal object interactions. This provides a comparison against non-target activity.

**X-CLIP inference**

The pretrained X-CLIP model compares the sampled video frames with the activity descriptions and produces similarity scores for the different classes. These scores are used to identify potential activity intervals.

**Temporal localization**

The scores are arranged chronologically to identify regions where an activity is likely to occur. Consecutive candidate windows are grouped, and the resulting intervals are used to estimate the activity's start and end times.

**VLM verification**

A vision-language model is used to examine candidate intervals in the original video. Its purpose is to determine whether the proposed activity is visually supported and provide additional context for the prediction.

The final output contains the predicted activity class, start time, end time and confidence score.

This revised approach separates initial activity recognition from contextual verification. X-CLIP provides the initial candidates, while the VLM helps assess whether those candidates are supported by the actual video.

## 3. Dataset and Evaluation

The current evaluation was conducted on a subset of 30 videos containing 35 ground-truth activity instances. The dataset included examples from multiple activity categories, including fighting, falling, fire/smoke, intrusion, vandalism, theft and abandoned objects.

The UCF-Crime categories were mapped to the closest corresponding assignment classes where appropriate. Additional falling and abandoned-object examples were included to cover activities not directly represented by the selected UCF-Crime categories.

The system was evaluated using two configurations:

- **Pure X-CLIP:** Predictions generated using X-CLIP without the additional VLM verification stage.
- **Full pipeline:** The revised pipeline incorporating X-CLIP candidate detection and VLM-based verification.

The evaluation considered whether the system identified the correct activities and whether its predicted temporal intervals overlapped with the ground-truth intervals.

The main metrics included target-anchored precision and F1-score, class-wise F1-score, temporal precision and recall at an IoU threshold of 0.5, and a more lenient vicinity-based temporal matching measure. Temporal boundary error was also recorded.

## 4. Results

### Quantitative results

The comparison between the two configurations showed a considerable increase in the number of detected activities after introducing the additional pipeline stages.

| Metric | Pure X-CLIP | Full pipeline |
|---|---:|---:|
| Videos evaluated | 30 | 30 |
| Ground-truth events | 35 | 35 |
| Predictions generated | 1 | 45 |
| Target-anchored hits | 1 | 11 |
| Target-anchored precision | 100% | 50% |
| Target-anchored F1-score | 6.45% | 42.31% |
| Macro F1-score | 4.76% | 36.51% |
| Temporal F1 @ tIoU 0.5 | 0% | 5% |
| Vicinity-based temporal F1 | 5.56% | 15% |
| Mean boundary error | 1.70 s | 2.16 s |

The pure X-CLIP configuration generated only one prediction across the evaluation videos. Although its reported anchored precision was 100%, its recall was extremely low. This shows that relying on X-CLIP alone with the current candidate-generation settings was too restrictive.

The full pipeline generated 45 predictions and identified 11 target-anchored hits. Its target-anchored F1-score increased from 6.45% to 42.31%, while macro F1 increased from 4.76% to 36.51%.

However, this improvement came with an increase in false positives. The full pipeline's anchored precision was 50%, indicating that a considerable number of its predictions did not correspond to the expected target activities.

Temporal localization also remains a major limitation. At a tIoU threshold of 0.5, the full pipeline achieved a temporal F1-score of only 5%. The vicinity-based F1-score was higher at 15%, suggesting that some predictions identified the correct general time region but did not accurately match the annotated boundaries.

### Class-wise observations

The reported class-wise results show different levels of performance across activities.

| Activity | Full pipeline F1-score |
|---|---:|
| Falling | 88.89% |
| Intrusion/Trespassing | 66.67% |
| Fire/Smoke | 60% |
| Theft/Chain Snatching | 40% |
| Fighting/Physical Assault | 0% |
| Vandalism | 0% |
| Abandoned Object | 0% |

Falling was the strongest class in this evaluation. Fire/Smoke and Intrusion/Trespassing also showed promising video-level detection results, although their temporal localization and false-positive performance still require improvement.

Fighting and Vandalism were not successfully detected according to the reported class-wise F1 results. This is particularly important because the system still struggles to identify specific physical actions and distinguish them from other human movements.

## 5. Qualitative Analysis

A closer examination of individual predictions helps explain the quantitative results.

**Successful falling detections**

In `Falling_fall-01-cam0-rgb.mp4`, the system predicted Falling between 0 and 6 seconds, while the ground-truth interval was 3 to 6.4 seconds. The prediction was supported by the VLM, which identified the person falling to the ground. Its temporal IoU was 0.469.

In `Falling_fall-02-cam0-rgb.mp4`, the predicted interval was 0 to 4 seconds, compared with the ground-truth interval of 1 to 4.4 seconds. This prediction achieved a temporal IoU of 0.682.

These examples show that the system can recognize a visible falling event. However, it tends to start the predicted interval earlier than the actual event.

**Correct activity recognition with inaccurate boundaries**

In `Falling_fall-05-cam0-rgb.mp4`, the system predicted Falling from 2 to 6 seconds, while the ground-truth event occurred from 5 to approximately 6 seconds. The VLM confirmed that a falling action was visible, but the predicted interval included several seconds before the actual event. The temporal IoU was only 0.248.

This illustrates an important distinction: recognizing the correct activity does not necessarily mean that its temporal boundaries are accurate.

**Confusion between activities in multi-stage incidents**

In `Fire_Smoke_Arson041_x264.mp4`, the system generated Intrusion/Trespassing predictions at 29–39 seconds and 67–77 seconds, even though the ground-truth target category was Fire/Smoke.

These predictions may represent actions occurring before the fire event. However, they should not automatically be treated as correct detections of Fire/Smoke. They also highlight the difficulty of interpreting different actions within a longer incident and distinguishing a related action from the actual target event.

**False positives and missed activities**

The confusion matrix indicates that Intrusion/Trespassing was a frequent spurious prediction, with 15 background instances assigned to this class. Fire/Smoke and Abandoned Object also received several spurious predictions.

On the other hand, several ground-truth events were missed completely, particularly Fighting, Vandalism and Abandoned Object.

This suggests that the current system still has difficulty distinguishing target activities from visually similar normal behaviour. It also indicates that the similarity scores and candidate-generation process need further refinement.

## 6. Conclusion and Future Improvements

The revised X-CLIP-based approach provides a more direct way of identifying activities than the initial description-based pipeline. Adding VLM verification increased the number of detected activities and improved the reported target-anchored F1-score.

However, the results also show that increasing the number of predictions does not automatically improve detection quality. False positives remain a significant problem, and temporal localization is still weak. Falling is currently the strongest class, while fighting, vandalism and abandoned-object detection need further work.

The next improvements will focus on refining activity prompts, particularly for difficult classes such as Fire/Smoke and Fighting/Physical Assault. Temporal window selection, candidate thresholds and boundary refinement also need further tuning. The VLM verification stage can be improved to reject unsupported predictions without removing genuine short-duration activities.

Overall, the revised pipeline is a useful starting point for zero-shot activity detection, but further work is required to improve its precision, recall and temporal localization before it can be considered reliable across all target activity classes.