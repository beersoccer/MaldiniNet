# MaldiniNet — A proprietary Neural

# Network for football match result

# prediction.

### Leveraging advanced expected goals metrics to better predict the

### 1x2 market for top european football leagues.

```
Paul Corcoran· Follow
Published inDev Genius · 17 min read · Aug 22, 2023
```
```
253 4
```
## Table of Contents

```
Introduction
Feature Exploration
Feature Engineering
Neural Network Creation
Match Prediction Process
Betting Strategy Simulation
Conclusion
```
## Introduction

Goals are like gold dust when it comes to a football match, for fans of
multiple sports a try or touchdown score is celebrated fondly, but arguably
not as joyful as a solidtary goal scored late in a 1–0 win in an important game
in a football match. Football is low scoring, most leagues will average
between 2.2–3 goals, if your unlucky you could witness many games over
your fandom where goals are non existent and games do not provide that one


euphoric blast of relief and happyness. But as a football fan that wont stop
you ever coming back for the next game because when those moments come
back they are truly special. Football, like any competitive sport is based on
scoring more than your opponent, having a better defence to keep goals out is
just as important as forward attacking play. Goals are random, we don’t ever
know when they will arrive. Everything about the game is stochastic but fans
and professional bettors will do everything they can to be in a better position
to forecast them and what teams will win.

Expected goals arrived as a metric in the mid 2010’s which aimed to better
explain the occurance of goals. Opta’s official definition captures the metric
perfectly.

```
Expected goals (or xG) measures the quality of a chance by calculating the
likelihood that it will be scored by using information on similar shots in the
past.
```
By classifying a shot on a measure between 0 and 1 the metric assigns a
probability on how likely that shot was expected to return a goal based on
several factors such as historical shots taken from that distance, the position
of the goalkeeper, the shot type, how many defenders were cutting off the
angle and so on. The level of features used to come up with the expected goal
probability will depend across services.

Expected goals are a good proxy for actual goals scored. They also provide
strong indicators for overall performance such as offensive/defensive and
chance creation power. Figure 1 shows the two metrics plotted together for
the dataset used for this Neural Network comprising of expected goals/goals
data for the top 5 european leagues. The top 5 leagues are the Premier
League, La Liga, Bundesliga, Ligue 1 and Serie A. Differences between the
two will fluctuate but I am in no doubt of its suitability of use for any
prediction model for football.

![image](https://github.com/user-attachments/assets/81c9cac5-dd9c-42d4-b44c-2c7933a35438)
```
Figure 1: Average xG versus Average Goals scored.
```
I always find football prediction an interesting topic, can a propriety model
created scraping and manipulating expected goals data become a good
predictor of future results? This is the aim and it is no doubt a fun exercise to
undertake while upskilling modeling skills at the same time.

The dataset used derives from a popular football statistics site understat.com.
I scraped each game to retrieve situational expected goals metrics, such as
the corner and open play expected goals. I estimated that by using more
illuminated expected goals features other than simple totals that didnt
explain where the expected goal came from would perform superior. I won’t
go into any detail here but if your interested in the data sourcing it was
published in an article I wrote here. The second sourced script extracts the
post win probabilities also scraped from understat, the post win xg
probabilities will serve as the target variables in a **_multivariate
regression problem_**. This article here demonstrates the sourcing of the
target variables. It is important to note that the target variables extracted are
estimates of these probabilities and not a total source of truth. But I do feel
that there is some value in using the post win probabilities after the game has
completed as a tool for future prediction.


This article will move forward and detail the **_feature exploration_** ,
some **_feature creation_** and **_engineering_** before detailing the steps and
procedures I took in **_creating_** the neural network amptly named for fun by
taking inspiration from one of the greatest defenders of all time Paolo
Maldini. (It seemed a cool name for the model aswell ⚽)

I will then **_validate_** my own method of prediction against the closing odds of
a subset of games in the 2nd part of last season 2022–23 in order to validate
any potential betting strategy that could be used. It will also give me ideas on
how best to use the predictions and further improve in iterations to come.
The code used for this article will be provided in full at the end so your
reading till the end will be appreciated.

## Feature Exploration

The number of rows for modeling was over 12k. The data set extracted
contains approximately 50 features pertaining to the league, season, home
and away team for each game played for the following metrics:

1. Direct Free Kick xG: From direct shots from dead ball situations.
2. Corner xG: Shots from corner situations.
3. Open Play xG: Shots from open play.
4. Penalty xG: Penalty kicks.
5. Set Piece xG: Shots deriving from free kicks that were not shot directly to
the goal.
6. Shots on target: The count of all shots on target deriving from the above
situations.
7. Shots: The count of all shots deriving from the above situations.
8. Goals: The count of all goals deriving from the above situations.


I plotted out the data set for all variables, noting the distinct presence of
outliers in the majority of features. Take the home_corner_xg variable where
the median was quite low around 0.15. The outliers demonstrate good
performance by a given team. How I deal with them is an important
consideration which I will touch on later.

![image](https://github.com/user-attachments/assets/4c93929e-c366-4106-9c75-a7e87d316b24)
```
Figure 2: Example Box Plots of the given variables.
```
One important action was to deal with the post win probabilities associated
with the draw_prob target variable. There were a small number of high draw
probabilities associated with some games. From experience a draw is roughly
priced between 20%-35% in a top level game. Having such noisy training data
was not desirable so these instances were removed.

![image](https://github.com/user-attachments/assets/54dc8873-170a-4d80-b059-7bd1c1e9be13)
```
Figure 3: Outliers in one of the target variables.
```

## Feature Engineering

The majority of feature engineering was completed during the intial data set
manipulated of shots data but I did want to strongly include on important
metric known to all fans in football — **home advantage.**

Home advantage, playing in front of a noisy stadium of supporters who want
the home team to win undoubtly has an effect in all sports odds pricing. Take
the next plot for example. If any data scientist looked at the line graph below
they would be instantly able to conclude that home teams tend to perform
better than away teams. The evidence for home advantage is extremely clear
and a model should take this into account.

![image](https://github.com/user-attachments/assets/e8dc629b-c08b-4b11-a2e2-a6b2dc65e612)
```
Figure 4: Home advantage.
```
To calculate home advantage I did a couple of steps. I first subtracted the
home_xG — away_xG fields. This were initially scraped but not included in
the features later as the other variables essentially consisted of the same
values just spread out over different situations.


I then calculated the average home advantage by team over each different
season. The idea was that teams change over time and that different home
teams have stronger home performances than others. The values were then
scaled and normalised to return more model friendly values and mapped
back to each team in the data set.

```python
# Load the stored DataFrame using pickle
with open('MaldiniNet.pickle', 'rb') as f:
df = pickle.load(f)
df['home_advantage'] = df['home_xG'] - df['away_xG']
# Calculate the mean home advantage per home team for each season
mean_home_advantages = df.groupby(['season', 'home_team'])['home_advantage'].mea
# Normalize home advantage values
scaler = StandardScaler()
normalized_home_advantages = scaler.fit_transform(mean_home_advantages['home_adv
# Create a mapping dictionary of (season, team) tuple to normalized home advanta
season_team_home_advantage_mapping = {
(season, team): normalized_home_advantages[i][0] for i, (season, team) in en
}
# Map normalized home advantage values back to the main DataFrame using (season,
df['home_advantage'] = df.apply(lambda row: season_team_home_advantage_mapping.g
```
To illustrate this better, lets take a look at one of the great Maldini’s old
teams AC Milan and show how their home advantage changed over the years
when they improved and became scudetto winners. We can see a clear
improvement over the years and then in the latter part of the data set when
they became champions their home advantage was at its highest. This metric
should add that little bit extra in the model’s outputs.

![image](https://github.com/user-attachments/assets/80b4152d-11ac-4322-83c3-748f795042d2)
```
Figure 5: Ac Milan’s Home advantage metric.
```
## Neural Network Creation

I used the subset of training data prior to 2023–01–01 for model training and
applied the train test split of 80/20 for the model training. The rest of the
data will be used to as validation set to test the outputs against real closing
odds shortly. There are several strong models available for this type of
prediction such as XGBoost and RandomForest but Neural Networks
potentially possess a stronger ceiling for what I am trying to do.

To further elobrate, Neural networks offer a compelling framework for
multivariate regression prediction tasks due to their inherent capacity to
model complex and interconnected relationships within data. In the context
of multivariate regression, where multiple input variables contribute to
predicting a continuous output, neural networks excel at capturing intricate
patterns, dependencies, and interactions among these variables. Through
layers of interconnected neurons, these networks learn to transform input
features into intermediate representations, gradually extracting and
combining relevant information.


We can think of Neural networks as a series of multiple regression models
which have weights associated with each neuron in the network which
ultimately has a say on the final output — the home win, draw win and away
win probabilities which to reiterate are the target variables in this
multivariate regression task. The inputs of a neural network should be scaled
in some manner prior to training.

The most important aspects of a neural network is as follows:

1. **Input Layer** : The input layer is where the network receives its initial
    data or features. Each node in this layer corresponds to a specific feature
    or input variable.
2. **Hidden Layers** : Hidden layers are intermediate layers between the
input and output layers. They play a crucial role in extracting and
transforming features from the input data. Neural networks with more
hidden layers are often referred to as “deep” networks, and this
architecture is known as a deep neural network (DNN).
3. **Output Layer** : The output layer produces the network’s final predictions
or results. The number of nodes in this layer depends on the nature of the
prediction task. For example, in a binary classification task, there might
be two output nodes representing the probabilities of the two classes. In a
regression task, there would be a single node for the continuous output.
4. **Connections (Synapses)** : Connections between nodes, known as
synapses, carry signals between layers. Each connection has a weight
associated with it, which determines the strength of the signal. During
training, these weights are adjusted to optimize the network’s
performance.
5. **Activation Functions** : Activation functions introduce non-linearity into
the network. Each node in a hidden layer applies an activation function to
the weighted sum of inputs. Common activation functions include ReLU
(Rectified Linear Unit), sigmoid, and softmax. Activation functions allow
neural networks to model complex relationships in the data.


I will provide the archecture of the MaldiniNet below.

```python
# Build the neural network model with L2 regularization
home_input = layers.Input(shape=(X_train_scaled.shape[1],), name='home_input')
away_input = layers.Input(shape=(X_train_scaled.shape[1],), name='away_input')
shared_layer1 = layers.Dense(12, activation='relu', kernel_regularizer=l2(0.01))
shared_layer2 = layers.Dense(6, activation='relu', kernel_regularizer=l2(0.01))
home_branch = shared_layer2(shared_layer1(home_input))
away_branch = shared_layer2(shared_layer1(away_input))
merged_branches = layers.concatenate([home_branch, away_branch])
output_layer = layers.Dense(3, activation='softmax')(merged_branches)
model = keras.Model(inputs=[home_input, away_input], outputs=output_layer)
```
```python
# Define the learning rate and create the optimizer
learning_rate = 0.
amsgrad = False
optimizer = Adam(learning_rate=learning_rate, amsgrad=amsgrad)
# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')
# Train the model
history = model.fit(
[X_train_scaled, X_train_scaled], y_train,
epochs=300, batch_size=16,
validation_data=([X_test_scaled, X_test_scaled], y_test),
callbacks=[early_stopping]
)
```
For this task, I created a neural network using two shared fully connected
layers, one for home team metrics and one for away team metrics running for
300 epochs. It takes input features related to both the home and away teams.
The architecture includes shared layers that extract relevant information
from the input features, followed by concatenation to merge these
representations. The output layer generates probabilities for three potential
outcomes (home win, draw, away win) using the softmax activation
function.Softmax normalizes the output scores so that they sum up to 1. This
is essential for interpreting the results as probabilities. It ensures that the


predicted probabilities are consistent and add up to 100%, making them
suitable for interpreting as win probabilities. L2 regularization is applied to
the shared layers to prevent overfitting by penalizing large weights. This
model structure enables the network to learn complex patterns and
dependencies in the input data. The learning rate of 0.001 allows the model
to learn slowly and Adam as the optimisation algorithm. Several variations of
all these parameters were ran to get to the parameters used.

Figure 5 is a proxy visualisation of the MaldiniNet, the first hidden layer is on
the left which then feeds into the next hidden layer and then finally the
output layer of 3 nodes which correspond to home win, draw and away win
probabilities.

![image](https://github.com/user-attachments/assets/5c390e7a-925d-46f4-8434-8a6c05144b08)
```
Figure 6: MaldiniNet composition of one branch.
```

Remember when I noted the presence of outliers earlier in the feature
exploration? I ran several models initially to analyse the evaluation metrics of
MSE wrt to the validation loss in the model performance. Finally setting on
the archecture and activation functions that seemed to not overfit the data.
When scaling the data prior to training I used two scaling techniques.
RobustScaler and MinMaxScaler.

Certainly! Robust Scaler and Min-Max Scaler are both preprocessing
techniques used to transform numerical features in order to prepare them for
machine learning algorithms. However, they differ in how they handle the
scaling and normalization process, making them suitable for different
scenarios. Some of the key differences are as follows:

**Robust Scaler:**

```
Scaling Approach: Robust Scaler scales features using the interquartile
range (IQR) and median, making it robust to outliers. It subtracts the
median and then divides by the IQR.
Outlier Sensitivity: Robust Scaler is designed to handle outliers well, as it
uses the median and IQR instead of mean and standard deviation, which
can be influenced by outliers.
Normalization Range: The scaling range depends on the IQR, which can
lead to a different scale compared to Min-Max Scaler.
Suitability: Robust Scaler is a good choice when your data contains
outliers or when you want to avoid outliers affecting the scaling process
significantly.
```
**Min-Max Scaler:**

```
Scaling Approach: Min-Max Scaler linearly scales features to a specific
range, usually between 0 and 1. It subtracts the minimum value and then
divides by the range (maximum — minimum).
```

```
Outlier Sensitivity: Min-Max Scaler is sensitive to outliers since the range
is influenced by extreme values. Outliers can distort the scaling process
and affect the transformed values.
Normalization Range: The normalization range is predefined (usually 0 to
1), which provides consistent scaling across different features.
Suitability: Min-Max Scaler works well when your data does not have
significant outliers and you want to maintain a consistent scaling range
for all features.
```
The presence of outliers in my dataset bar the draw win target variable was
deemed not to be a negative. The outliers themselves represented key data
points about a teams strengths in each situation and I felt the model would
do better by applying the MinMaxScaler which perserved these data points.
To further cement my decision I plotted the loss curves of both models, one
for Robust scaling and one for MinMax scaling.

![image](https://github.com/user-attachments/assets/8a9b74bd-0dca-492d-a171-7b443800573f)
```
Figure 7: RobustScaling loss curves.
```
By evaluating the plots of both techniques, they are both very close. Robust
scaling does not overperform the MinMaxScaling to such an extent that I can
conclude that it is the better approach. Not to mention I feel as though I am
losing vital information by removing these supposed outliers. I decide to use
the latter going forward. I will also mention that the loss values are very low,
which initially was a concern as this could potentially mean the model is


learning to replicate the values too closely and will fail to generalise to new
predictions. The training loss values were slightly higher than the validation
loss values which is always a good sign. The only way I could prove that the
model could function as a good predictor of future match probabilities was to
validate this against real life closing odd values and simulate a betting
strategy at different thresholds.

![image](https://github.com/user-attachments/assets/8503df4d-2371-47a2-979f-4ed142cf20ef)
```
Figure 8: MinMaxScaling loss scurves.
```
## Match Prediction Process

I use the rolling 6 game averages for home team and away team as inputs.
The thinking here is that team performance is comparable to a time series
analysis. That is that performances change over time and the performance of
a team at the start of the season in August should not be totally predictive of
their potential performance in January the following year. By taking the
rolling 6 game averages this allows for form to be captured as well as
contrasting styles against specific teams at least in theory. To start the rolling
window off I used the last 6 games averages in the training set as the first
game for each team in the validation set would have no inputs, then as time
goes on they will start to record new averages for games played in early 2023.
I should evaluate a hypothetical betting strategy in order to ascertain if the
model is actually any good at predicting 1x2 markets.


#Calculate rolling averages for each home team and away team in the validation s
rolling_window_size = 6

validation_rolling_home_averages = []
validation_rolling_away_averages = []

for i in range(len(validation_data)):
validation_match = validation_data.iloc[i]
home_team = validation_match['home_team']
away_team = validation_match['away_team']

# Get the last 6 games for the current home team and away team from the trai
home_window_data = training_data[training_data['home_team'] == home_team].ta
away_window_data = training_data[training_data['away_team'] == away_team].ta

if len(home_window_data) < rolling_window_size:
# Extend the window using training data until 6 games are available
missing_games = rolling_window_size - len(home_window_data)
home_window_data = pd.concat([training_data[training_data['home_team'] =
if len(away_window_data) < rolling_window_size:
# Extend the window using training data until 6 games are available
missing_games = rolling_window_size - len(away_window_data)
away_window_data = pd.concat([training_data[training_data['away_team'] =

# Calculate the rolling average for home team and away team
rolling_home_average = home_window_data[input_features].mean(axis= 0 )
rolling_away_average = away_window_data[input_features].mean(axis= 0 )

validation_rolling_home_averages.append(rolling_home_average)
validation_rolling_away_averages.append(rolling_away_average)

# Convert the lists of rolling averages to numpy arrays
validation_rolling_home_averages = np.array(validation_rolling_home_averages)
validation_rolling_away_averages = np.array(validation_rolling_away_averages)

# Scale the rolling averages using the same scaler used for training data
validation_rolling_home_averages_scaled = scaler.transform(validation_rolling_ho
validation_rolling_away_averages_scaled = scaler.transform(validation_rolling_aw

# Make predictions on the scaled rolling averages using the trained model
validation_predictions = model.predict([validation_rolling_home_averages_scaled,

# Add the predicted probabilities to the validation data DataFrame
validation_data['pred_home_win'] = validation_predictions[:, 0 ]
validation_data['pred_draw'] = validation_predictions[:, 1 ]
validation_data['pred_away_win'] = validation_predictions[:, 2 ]


## Betting Strategy Simulation

I intentionally held out some data (799 games) to test my predicted outputs
against the real life closing odds that the bookmakers offered. The data in the
validation set for betting simulation comprised of all games for each league in
January 2023 to the end of the season in June 2023.Bookmakers existence
depends on getting the odds correct more often than not for top level games
so they will act as the closest source of truth available. I extracted these odds
from an odds comparison site and removed the margin. This was important
as the model I created outputted theoretical 100% odds with no margin
applied. Margin is applied by bookmakers to overestimate the likelihood of a
team winning and thus creates the mathematical edge on which their abilility
to make money exists. A model should aim to beat this margin at the very
least to generate long term profits. The file ClosingOdds.py contained in the
github repo performs this extraction and margin removal to estimate 100
percent odds. I compared the percentages of my predicted outcomes with the
100 percent percentages for each game and defined a number of thresholds
to bet. Remember odds are just percentages expressed by 1/odds percentage.

For example, if my model said that team A at home was 0.51 percent chance
to win and the closing odds with margin removed suggested a 0.50 percent
chance of winning I would place a 10 euro bet on team A. I included the result
in this validation file which would allow me to calculate the ROI of each
threshold. I did this at various thresholds across the home/draw/away win
markets. The returns would be dictated by the 100 percent odds. If the home
team did win then it would be 10 x 100%Odds and -10 for any other result.

Figure 8 shows the results of betting home teams when the predicted home
win from my model differed from the 100% percentages offered by
bookmakers. The results are good. While you can never conclude that yes this
model is a perfect predictor and sure fire way to make money it does mean
that over a decent size sample (potential 799 games to bet on) that the model
could potentially be fit for purpose. As I higher the thresholds for differences
between predicted odds percentages against closing odds percentages the bet


count lowers, when the model seems to disagree with the bookmaker odds at
the extreme end of 20% the ROI is very good. Being the sceptical person I am
when it comes to a model (especially one like this) I would not look at this
and think my model is perfect. There are naturally a few reasons why the
team was that price, team news, form and ultimately sharp money dictated
their closing odds. I will categorically state that no model predicting in the
top 5 leagues should disagree by anything over 10%, these games have the
highest price confidence.

![image](https://github.com/user-attachments/assets/c6438a8e-d426-4aab-8b9a-14db0a478a6f)
```
Figure 9: Home team betting simulation at different thresholds.
```
Betting the draw is not something typical of sharp betting, who turn to asian
handicap markets in order to get an underdog on side. The model again
potentially becomes valuable blind betting the draw at the lower end
thresholds. The higher and totally unrealistic thresholds suffer from minor
losses.

![image](https://github.com/user-attachments/assets/51e64da8-d257-46a7-aca8-82701ccdfb7d)
```
Figure 10: Draw win betting simulation at different thresholds.
```

Betting on the away team where there is a different of opinion between my
predicted odds and closing odds now shows the model in a different light.
Only one threshold is actually positive ROI. Bet count is much lower than the
other two markets which is a positive, I can assume that it is taking in home
advantage when deciding on the odds outcomes. If I had of trusted the model
wrt betting the away team I would have no doubt lost money. This illustrates
why any model should be backtested and simulated before even attempting to
use.

![image](https://github.com/user-attachments/assets/40fd7ec6-ecc2-4deb-9c48-8f2027a14897)
```
Figure 11: Away win betting simulation at different thresholds.
```
## Conclusion

Predicting football match outcomes is no easy task. No model will ever be
able to incorporate key information should as injuries/different personal on
the pitch and other aspects of the beautiful game that the most powerful
models in the world can. Even saying that the models that bookmakers use
incorporate one massive aspect — real life money which shapes and moves
prices over market maturation right up until the kick off. I do think that a
combination of a powerful model such as a neural network created here and
strong domain knowledge can act as a dual ensemble prediction model that
can guide an individual in making money off the top level of football but it
takes time and strong data. I will continue to assess the training and potential
avenues for more data involvement in this personal project which I have to
admit is a fun task albeit not without its frustrations :). Perhaps more data
involving other match statistics could help improve prediction power. I think


that game state is an important feature when it comes to expected goals
creation and I will seek to create an even more powerful version of the data
set in the near future, other relevant information would be actual form wrt
actual results. If you have read up until now you have my thanks and
congratulations. The code used to create the V1 of the MaldiniNet can be
found on my github here. It is by no means a near finished model and I will
continue to expand and improve in the coming months.

```
Get an email whenever Paul Corcoran publishes.
Get an email whenever Paul Corcoran publishes. By signing up,
you will create a Medium account if you don't already...
footballdotpy.medium.com
```
If you are interested in football and python and dont already follow me, I
would appreciate your support to my work. I will update this project in the
coming months and hopefully you may have gained some knowledge or
coding inspiration for your own models.

```
Soccer Soccer Analytics Football Betting Football Neural Networks
```
