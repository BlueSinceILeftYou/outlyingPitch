# Baseball Project Ideas

## Baseball Databank Ideas
Potential ideas:
- Generate fake baseball players and assign various fields to them (school they attended, age, height/weight, batting avg etc) and predict any of the following:
  - success in their careers
  - team they will be traded to next (if any)
  - lifetime award count
  - total revenue earned
  - hall of fame status

---

## Baseball Idea: Pitcher Classification and Outlier Detector
### Why?
Recently, a few pitchers have gotten in trouble for throwing uncharacteristic pitches in certain counts, and they were then traced to bets. We could take a pitch and compare it to a pitcher’s previous pitches. This could detect how strange a pitch was.

### How?
- Gather data from MLB savant using pybaseball.  
- Clean data to sort by pitcher.  
- Train many binary classifiers for who threw a given pitch.  
- There are multiple ways we can do this with multi-classification.  
- We have access to many features for each pitch, like spin axis, count it was thrown in, release point, velocity, etc.  

### Models We Could Try
- XGBoost  
- Random Forest  
- Logistic Regression  
- CatBoost  

We could also use these classifiers to compare pitcher similarity, examining the linear weights of pitcher classifiers against each other.

### Limitations
- Visualizing the data might be annoying.  
- For each classifier, most of the training data is from one category (not thrown by the pitcher), need to make sure our model is differentiating and isn’t just outputting 0.  

---

## Other Project Directions
Honestly cool with doing this kind of idea with any type of dataset:
- Predict the prevalence/mortality of a fake disease with random characteristics based on the data of previous diseases: 1, 2, 3  
- Predict the impact of an arbitrary legislative change based on the available economic data from previous decisions and estimated conditions at that time vs this time: link  
- Predict future behavior in animal populations based on current known trends: 1, 2, 3  

https://x.com/CreatureOnBased/status/1914138585550168398

---

## Small Ideas
- Swing Mechanics Classifier: OpenBiomechanics + Swing keypoint  
- Pitch type & speed prediction from video clips: MLB YouTube dataset  
- Real-time object tracking: ball + bat or hands  
- Comparing pose estimation models for baseball swings: baseballcv  

---

## Pitcher Classification and Outlier Detector
### Goal
Given a pitch and relevant features, successfully predict which pitcher threw it, to detect outliers and possibly suspicious pitches.  

### Collected Data
Using MLB savant data pulled from pybaseball, we have access to these features for nearly every pitch thrown in the major leagues:  
https://baseballsavant.mlb.com/csv-docs  

We pull the data using pybaseball, then clean it down to the relevant features, and finally organize pitches by pitcher.  

### Modelling
We could use one of two approaches, or both:
- Unsupervised learning for pure outlier detection.  
- Classifiers for each pitcher using XGBoost, CatBoost, logistic regression, etc. and choosing the best model.  

We then consider a pitch “suspect” if the model decides it is unlikely that the pitcher threw it.  

### Visualization
Need to think about this more. We could make:
- A pitcher similarity thing?  
- A histogram of how correct the model was?  
- Confusion matrices?  

### Testing
Unsure, probably an 80% train / 20% test split.  
It’s hard because if we’re making binary classifiers, a vast majority of the data will be for the negative case (pitches not thrown by the pitcher).  
